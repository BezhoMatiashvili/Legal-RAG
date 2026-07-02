from datetime import date


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


def generate_start_url_batches(start_date: date, end_date: date) -> tuple[list[str], list[str]]:
    start_date_str = start_date.strftime(DATE_FORMAT)
    end_date_str = end_date.strftime(DATE_FORMAT)
    first_batch = []
    deferred_batch = []

    for topic in TOPICS:
        for additional_status in ADDITIONAL_STATUSES:
            url = START_TEMPLATE.format(start_date_str, end_date_str, topic, additional_status)
            if topic == "" or additional_status == "":
                deferred_batch.append(url)
            else:
                first_batch.append(url)

    return first_batch, deferred_batch


def generate_start_urls(start_date: date, end_date: date) -> list[str]:
    first_batch, deferred_batch = generate_start_url_batches(start_date, end_date)
    return [*first_batch, *deferred_batch]
