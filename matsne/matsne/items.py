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
