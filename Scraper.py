import time
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime as dt
from collections import namedtuple
from playwright.sync_api import sync_playwright

URLS_PER_PAGE = 50

ScrapeInfo = namedtuple(
    "ScrapeInfo",
    ["browser_page", "decade", "genre", "save_every", "data", "num_scraped"],
)

decades = [1950 + 10 * i for i in range(8)]
genres = {
    "Rock": 4,
    "Metal": 8,
    "Pop": 14,
    "Folk": 666,
    "Country": 49,
    "Soundtrack": 680,
    "R&B, Funk & Soul": 1787,
    "Religious Music": 1016,
    "Electronic": 16,
    "Hip Hop": 45,
    "World Music": 195,
    "Classical": 216,
    "Jazz": 84,
    "Reggae & Ska": 1781,
    "Blues": 99,
    "Comedy": 79,
    "Disco": 85,
    "New Age": 695,
    "Experimental": 667,
    "Darkwave": 211,
}


def wait_for_multiple_selectors(page, selectors, timeout=30):
    start_time = time.time()

    while time.time() - start_time < timeout:
        for selector in selectors:
            if page.query_selector(selector):
                return selector
        time.sleep(0.1)

    raise TimeoutError(
        f"None of the selectors {selectors} were found in {timeout} seconds."
    )


def search_url(decade, page, genre=None):
    if genre:
        return f"https://www.ultimate-guitar.com/explore?genres[]={genres[genre]}&decade[]={decades[decade]}&order=rating_desc&page={page}&type[]=Chords"
    else:
        return f"https://www.ultimate-guitar.com/explore?decade[]={decades[decade]}&order=rating_desc&page={page}&type[]=Chords"


def get_search_info(decade, page, browser_page, genre=None):
    """Get the URLs, ratings and stars of all songs on a page."""
    browser_page.goto(search_url(decade, page, genre), wait_until="domcontentloaded")

    wait_for_multiple_selectors(
        browser_page, [".LQUZJ", "h1 >> text='Oops! We couldn't find that page.'"]
    )

    html = browser_page.content()
    soup = BeautifulSoup(html, "html.parser")

    urls, ratings, stars = [], [], []

    for div in soup.find_all("div", class_="LQUZJ"):
        a = div.find("a", class_="HT3w5", href=True)
        if not a:
            continue
        urls.append(a["href"])

        try:
            ratings.append(div.find("div", class_="djFV9").text.replace(",", ""))
        except:
            ratings.append("NaN")

        try:
            star_count = 5.0
            star_parent = div.find("div", class_="NlkcU")
            for star in star_parent.find_all("span"):
                star_classes = star.get("class", [])
                if "H3fQr" in star_classes:
                    star_count -= 1
                elif "RCXwf" in star_classes:
                    star_count -= 0.5
            stars.append(str(star_count))
        except:
            stars.append("NaN")

    return urls, ratings, stars


def get_song_info(url, browser_page):
    """Get the title, artist, chords and chord map of a song given its URL."""
    browser_page.goto(
        url,
        wait_until="domcontentloaded",
    )

    chord_map = {}

    def handle_route(route, request):
        route.continue_()
        if "/piano-inversions" in request.url:
            response = request.response().json()
            for symbol, chord_obj in response["info"].items():
                if chord_obj != []:
                    for key, value in chord_obj.items():
                        chord_map[key] = value["notes"]

        if "/transpose" in request.url:
            response = request.response().json()
            for symbol, chord_obj in response["info"]["applicature"].items():
                chord_map[symbol] = chord_obj["0"]["notes"]

    browser_page.route("**/*", handle_route)

    browser_page.wait_for_selector("h1.dUjZr")

    html = browser_page.content()
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    try:
        title = soup.find("h1", class_="dUjZr").text[:-7]
    except:
        print("Error on title")
    artist = ""
    try:
        artist = soup.find("a", class_="aPPf7 fcGj5").text
    except:
        print("Error on artist")
    chords = ""
    for chord in soup.find_all("span", class_="fciXY"):
        chords += chord.text + " "

    browser_page.click(".rPQkl.yDkT4.gm3Af.lTEpj.qOnLe >> text='Piano'")
    browser_page.wait_for_selector("header.Ufuqr > span:not(:empty)")
    browser_page.wait_for_timeout(400)

    return title, artist, chords, chord_map


def checkpoint(data):
    """Save a checkpoint of the data to a CSV."""
    filename = f"Checkpoints/checkpoint{dt.now().strftime('%Y-%m-%d %H-%M-%S')}.csv"
    pd.DataFrame(data).to_csv(filename, index=False)


def scrape_song(info, url, ratings, stars, index):
    """Scrape details of a single song and update the data list."""
    title, artist, chords, chord_map = get_song_info(url, info.browser_page)
    info.data.append(
        {
            "url": url,
            "title": title,
            "artist": artist,
            "decade": decades[info.decade],
            "genre": info.genre,
            "ratings": ratings[index],
            "stars": stars[index],
            "chords": chords,
            "chord_map": chord_map,
        }
    )
    info = info._replace(num_scraped=info.num_scraped + 1)

    if info.num_scraped % info.save_every == 0 and info.num_scraped != 0:
        print(f"{dt.now()} Scraped {info.num_scraped} songs")
        checkpoint(info.data)

    return info._replace(data=info.data, num_scraped=info.num_scraped)


def scrape_page(info, page, url_from, url_to):
    """Scrape all songs from a page and update the data list."""
    urls, ratings, stars = get_search_info(
        info.decade, page, info.browser_page, genre=info.genre
    )

    prev_pages_urls = (page - 1) * URLS_PER_PAGE
    start_url_on_page = url_from - prev_pages_urls
    end_url_on_page = url_to - prev_pages_urls

    for index, url in enumerate(
        urls[start_url_on_page:end_url_on_page], start_url_on_page
    ):
        try:
            info = scrape_song(info, url, ratings, stars, index)
        except Exception as error:
            print(f"Error on {url}: {error}")
    return info._replace(data=info.data, num_scraped=info.num_scraped)


def get_data(browser_page, url_from, url_to, save_every):
    """Scrape songs from multiple pages and save the data to a CSV.

    Args:
        browser_page: A Playwright browser page.
        url_from (inclusive): For every page, scrape songs starting from this URL.
        url_to (exclusive): For every page, scrape songs up to this URL.
        save_every: Save a checkpoint every time this many songs are scraped.
    """
    print(f"{dt.now()} Starting...")
    data = []
    num_scraped = 0
    info = ScrapeInfo(
        browser_page=browser_page,
        decade=None,
        genre=None,
        save_every=save_every,
        data=data,
        num_scraped=num_scraped,
    )

    for decade in range(len(decades)):
        info = info._replace(decade=decade)
        for genre in list(genres.keys()):
            info = info._replace(genre=genre)
            start_page = url_from // URLS_PER_PAGE + 1
            end_page = (url_to - 1) // URLS_PER_PAGE + 2
            for page in range(start_page, end_page):
                try:
                    info = scrape_page(info, page, url_from, url_to)
                except Exception as error:
                    print(f"Error on page {page}: {error}")

    print(f"{dt.now()} Done!")
    pd.DataFrame(info.data).to_csv(f"data_{url_from}-{url_to-1}.csv", index=False)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # Visible for debugging
    context = browser.new_context(**p.devices["Desktop Chrome"])
    page = browser.new_page()

    # Change the arguments according to your needs
    get_data(page, url_from=0, url_to=100, save_every=100)
