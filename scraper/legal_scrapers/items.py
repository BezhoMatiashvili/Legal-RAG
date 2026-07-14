# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import MapCompose, TakeFirst

from .utils.markdown import html_to_markdown


def class_to_status(class_str: str) -> str | None:
    if 'panel-info' in class_str:
        return 'ასამოქმედებელი აქტები'
    if 'panel-success' in class_str:
        return 'ძალაში მყოფი აქტები'
    if 'panel-danger' in class_str:
        return 'ძალადაკარგული აქტები'



class MatsneItem(scrapy.Item):
    # Existing fields
    document_url = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    document_id = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    language = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())

    title = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    document_number = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    document_recipient = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    adoption_date = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    document_type = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    document_topic = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    registration_code = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())

    body_markdown = scrapy.Field(input_processor=MapCompose(html_to_markdown), output_processor=TakeFirst())

    publication_source = scrapy.Field(input_processor=MapCompose(lambda s:s.rpartition(', ')[0] ,str.strip), output_processor=TakeFirst())
    publication_date = scrapy.Field(input_processor=MapCompose(lambda s:s.rpartition(', ')[-1] , str.strip), output_processor=TakeFirst())

    consolidated_publications = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())

    entry_into_force_date = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    expiry_date = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())

    status = scrapy.Field(input_processor=MapCompose(str.strip, class_to_status), output_processor=TakeFirst())
    additional_status = scrapy.Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())

    # Consolidation metadata. is_consolidated = "matsne main (consolidated) document"
    # (matsne's type=main class: the base act carrying the current consolidated text, as
    # opposed to amendment/informational acts). Derived from the detail page's
    # #publication-switcher (≥1 listed version ⇒ True — sound but incomplete: a base act
    # never amended has no switcher) and forced True by provenance in doc_type=main
    # crawls. consolidated_count = number of switcher versions (0 = never amended). Set
    # directly on the loaded item in parse_document (not via a processor), plain fields.
    is_consolidated = scrapy.Field()
    consolidated_count = scrapy.Field()
    consolidated_dates = scrapy.Field()


def _strip(value):
    """Strip strings, pass through non-strings (ints/None/etc.) unchanged.

    The new spiders feed ItemLoaders values straight from JSON, so the strip
    processor must tolerate non-string values that ``str.strip`` would choke on.
    """
    return value.strip() if isinstance(value, str) else value


def _f() -> scrapy.Field:
    """A trimmed, single-value field (the common case for the new spiders)."""
    return scrapy.Field(input_processor=MapCompose(_strip), output_processor=TakeFirst())


def _body() -> scrapy.Field:
    """The document body. Already converted to Markdown by the spider, so take it
    as-is (don't strip internal whitespace)."""
    return scrapy.Field(output_processor=TakeFirst())


def _list() -> scrapy.Field:
    """A structured (list/dict) field. The spider assigns it directly on the loaded
    item rather than through an ItemLoader, so no per-element processing is applied."""
    return scrapy.Field()


class EcdItem(scrapy.Item):
    """ecd.court.ge — Electronic Court Decisions (JSON API)."""

    source_url = _f()
    document_id = _f()              # "Id", e.g. "1-5861405"
    decision_document_id = _f()     # numeric DecisionDocumentId
    instance_id = _f()
    instance_name = _f()
    case_id = _f()
    case_no = _f()
    court_name = _f()
    case_category_name = _f()
    decision_type_name = _f()
    litigation_type_name = _f()
    decision_date = _f()            # ISO YYYY-MM-DD (converted from /Date(ms)/)
    barcode = _f()
    body_markdown = _body()


class ConstcourtItem(scrapy.Item):
    """constcourt.ge — Constitutional Court judicial acts (server-rendered HTML)."""

    source_url = _f()
    legal_id = _f()
    title = _f()
    doc_type = _f()
    number = _f()
    date = _f()
    publication_date = _f()
    authors = _f()
    college = _f()
    docx_url = _f()
    source_binary_url = _f()
    content_kind = _f()
    content_complete = _f()
    extraction_status = _f()
    body_markdown = _body()


class NaprItem(scrapy.Item):
    """napr.gov.ge — Public Registry legal-practice decisions (JSON list + PDF body)."""

    source_url = _f()
    document_id = _f()             # LETTERS_ID
    app_no = _f()                  # RANDOMID
    date = _f()                    # REGISTRATIONDATE
    sender = _f()                  # SENDER (already masked by the source)
    title = _f()                   # ABOUT
    dispute_category = _f()        # "დავის ტიპი/კატეგორია" search filter (ptag)
    decision_type_name = _f()      # Derived from the leading title phrase
    decision_date = _f()           # KANC_DATE
    decision_no = _f()             # KANC_NO
    pdf_url = _f()
    source_binary_url = _f()
    content_kind = _f()
    content_complete = _f()
    extraction_status = _f()
    body_markdown = _body()


class TbappealItem(scrapy.Item):
    """tbappeal.court.ge — Tbilisi Court of Appeals decisions (server-rendered HTML)."""

    source_url = _f()              # detail URL, also the dedupe key
    slug = _f()
    title = _f()
    date = _f()
    pdf_url = _f()
    source_binary_url = _f()
    featured_image_url = _f()
    article_summary = _body()
    content_kind = _f()
    content_complete = _f()
    extraction_status = _f()
    body_markdown = _body()


class SupremecourtItem(scrapy.Item):
    """supremecourt.ge — Supreme Court cases (HTML served by the /ka/getCases AJAX)."""

    source_url = _f()
    case_id = _f()
    chamber = _f()                 # official chamber name
    case_number = _f()
    date = _f()
    subject = _f()
    result = _f()
    appeal_type = _f()
    docx_url = _f()
    body_markdown = _body()


class TasItem(scrapy.Item):
    """tas.ge / docs.tbilisi.gov.ge — Tbilisi Architecture Service documents (ExtJS/DWR).

    The list crawl (``getDocsForPublicInfo``) provides the base fields; each document is
    then enriched from the public detail payload (``getUserDocumentLastMotion``), which
    carries the applicant, cadastral/owner info, request text, attachments, the answers/
    decision numbers, and the full decision text. Detail fields are ``None`` when the
    detail fetch fails or the document has not yet been submitted.
    """

    # --- list-level fields ---------------------------------------------------
    source_url = _f()
    document_id = _f()
    document_no = _f()
    address = _f()
    registration_date = _f()
    create_date = _f()
    status = _f()
    nomenclature = _f()
    nomenclature_case_id = _f()
    content_kind = _f()
    content_complete = _f()
    extraction_status = _f()
    body_markdown = _body()

    # --- detail: document / decision ----------------------------------------
    document_type_id = _f()
    deadline_date = _f()            # პასუხის გაცემის ვადა (deadLineDate)
    acquaint_date = _f()            # გაცნობის თარიღი (motion.whenUserOpenedTheDoc)
    decision_status_id = _f()       # document.documentStatusId
    decision = _f()                 # გადაწყვეტილება, e.g. "თანხმობა" (STATUSES map)
    decision_no = _f()              # გადაწყვეტილება № (latest response previousMotionId)
    can_see_final_result = _f()
    amount_to_pay = _f()
    response_markdown = _body()     # document.responseText (HTML) -> Markdown

    # --- detail: request / form ---------------------------------------------
    request_text = _body()          # მოთხოვნის ტექსტი
    form_fields = _list()           # [{label, value}] from docValues + fieldsetPojos
    attachments = _list()           # [file names]
    nomenclature_full = _f()        # nomenklaturMarkup -> text

    # --- detail: applicant (docAuthor) --------------------------------------
    applicant_name = _f()           # განმცხადებელი
    applicant_first_name = _f()
    applicant_last_name = _f()
    applicant_personal_no = _f()
    applicant_birth_date = _f()
    applicant_address = _f()
    applicant_email = _f()
    applicant_phone = _f()
    applicant_passport = _f()
    applicant_person_id = _f()

    # --- detail: executor (executorEmployee) --------------------------------
    executor_name = _f()
    executor_personal_no = _f()
    executor_email = _f()
    executor_phone = _f()
    executor_id = _f()

    # --- detail: cadastral / parcels (mapInfos) -----------------------------
    parcels = _list()               # [{cad_code, address, area, purpose, owner, coowner, reg_no}]
    cad_code = _f()                 # primary parcel, flattened
    land_area = _f()
    land_purpose = _f()
    owner = _f()
    coowner = _f()

    # --- detail: answers (oldResponseMotions) -------------------------------
    responses = _list()             # [{decision_no, acquaint_date, motion_date, status_id}]
