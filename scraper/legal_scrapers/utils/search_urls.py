from datetime import date, timedelta


TOPICS = [
    "კონსტიტუციური კანონმდებლობა, სახალხო დამცველი, მოქალაქეობა, ლტოლვილები, არჩევნები, რეფერენდუმი, პროფესიული კავშირები, პრესა, პოლიტიკური გაერთიანებები",
    "თავდაცვის, უშიშროების, საზოგადოებრივი წესრიგის შესახებ",
    "სოფლის მეურნეობა",
    "ბიუჯეტი, საბანკო, საფინანსო საკითხები და გადასახადები, ლიცენზიები და ნებართვები, სახელმწიფო შესყიდვები, აუდიტორული საქმიანობა",
    "ეკონომიკა, მრეწველობა, ვაჭრობა, ტრანსპორტი, კავშირგაბმულობა, მშენებლობა, საგზაო, რეგიონალური განვითარება, ტექნიკური რეგლამენტები",
    "საკუთრება, საჯარო რეესტრი და მიწით სარგებლობა",
    "სისხლის სამართალი, სასჯელაღსრულება",
    "განათლება, მეცნიერება, საინფორმაციო ტექნოლოგიები, კულტურა, სპორტი და ტურიზმი",
    "მართვა და სახელმწიფო ხელისუფლების ორგანოები",
    "სამოქალაქო კანონმდებლობა, ნოტარიატი, გაკოტრება, სამეწარმეო, პრივატიზაცია, სააღსრულებო, საარქივო საქმიანობა, საზოგადოებრივი გაერთიანებები",
    "ადმინისტრაციული საკითხები",
    "სასამართლო სისტემა, ადვოკატურა",
    "ჯანმრთელობის დაცვა, დაზღვევა, შრომა, სოციალური საკითხები",
    "ადგილობრივი თვითმმართველობა",
    "კოდექსები",
    "გარემო, ბუნებრივი რესურსები, ენერგეტიკა, ნავთობი, გაზი, წყალმომარაგება",
    "საგარეო და საერთაშორისო ურთიერთობები",
    "ადგილობრივი თვითმმართველობის ბიუჯეტი, ფინანსური, სოციალური საკითხები",
    "",
]
ADDITIONAL_STATUSES = [
    "ნორმატიული",
    "ინფორმაციული",
    "",
]

# TODO What if topic or status is None

START_TEMPLATE = "https://matsne.gov.ge/ka/document/search?publishing_date_fr%5Bdate%5D={}&publishing_date_to%5Bdate%5D={}&type=all&page=1&limit=100&label={}&additional_status={}"
DATE_FORMAT = "%d-%m-%Y"


def first_qs_value(params: dict, key: str) -> str | None:
    values = params.get(key)
    return values[0] if values else None


def sub_windows(start: date, end: date, granularity: str) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into contiguous, non-overlapping, gap-free sub-windows.

    matsne's ``document/search`` filters by publication date; a single (topic × status)
    listing over the whole ~1900→today range paginates ~1,500 pages, and one dropped
    "შემდეგი" link silently loses the tail. Splitting the range into shallow sub-windows
    (``"yearly"`` / ``"monthly"``) caps each listing's depth so no window can hide docs.
    The first/last windows are clamped to the real bounds, so coverage is exact.
    """
    if start > end:
        return []
    windows: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        if granularity == "yearly":
            nxt = date(cur.year + 1, 1, 1)
        elif granularity == "monthly":
            nxt = date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        else:
            raise ValueError(f"unknown granularity {granularity!r}")
        windows.append((cur, min(end, nxt - timedelta(days=1))))
        cur = nxt
    return windows


def build_search_url(start: date, end: date, topic: str, additional_status: str) -> str:
    """One ``document/search`` listing URL (page 1) for a (window × topic × status) cell."""
    return START_TEMPLATE.format(
        start.strftime(DATE_FORMAT), end.strftime(DATE_FORMAT), topic, additional_status
    )


def generate_start_url_batches(start_date: date, end_date: date) -> tuple[list[str], list[str]]:
    """Search-listing seed URLs, split into yearly windows for shallow, robust pagination.

    Returns ``(first_batch, deferred_batch)``: the deferred (catch-all) batch is every cell
    where ``topic`` or ``additional_status`` is empty — that catch-all alone enumerates the
    whole corpus, so it is drained last (phase 2). Windowing multiplies each (topic × status)
    cell by the yearly sub-windows of ``[start_date, end_date]``; the bucket rule is unchanged.
    """
    first_batch = []
    deferred_batch = []

    for topic in TOPICS:
        for additional_status in ADDITIONAL_STATUSES:
            for win_start, win_end in sub_windows(start_date, end_date, "yearly"):
                url = build_search_url(win_start, win_end, topic, additional_status)
                if topic == "" or additional_status == "":
                    deferred_batch.append(url)
                else:
                    first_batch.append(url)

    return first_batch, deferred_batch


def generate_start_urls(start_date: date, end_date: date) -> list[str]:
    first_batch, deferred_batch = generate_start_url_batches(start_date, end_date)
    return [*first_batch, *deferred_batch]
