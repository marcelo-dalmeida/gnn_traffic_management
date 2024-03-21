import copyreg
from functools import lru_cache

import lxml.etree as etree


copyreg_registered = False

parser = etree.XMLParser(remove_blank_text=True)


@lru_cache(maxsize=None)
def parse_xml(file, is_error_ok=False):

    try:
        xml = etree.parse(file, parser)
    except Exception as e:
        if is_error_ok:
            return None
        else:
            raise e

    return xml


def write_xml(root, file):
    et = etree.ElementTree(root)
    et.write(file, pretty_print=True)


def etree_unpickler(data):
    return etree.fromstring(data)


def etree_pickler(tree):
    data = etree.tostring(tree)
    return etree_unpickler, (data,)


def register_copyreg():
    global copyreg_registered

    if not copyreg_registered:
        copyreg.pickle(etree._ElementTree, etree_pickler, etree_unpickler)
        copyreg.pickle(etree._Element, etree_pickler, etree_unpickler)
        copyreg_registered = True


def rename_xml_string(element, old_string, new_string, parser=None):
    if parser is None:
        parser = etree.XMLParser(remove_blank_text=True)

    # method 'c14n2' doesn't add namespace to xml
    string_content = etree.tostring(element, method='c14n2')

    string_content = string_content.decode().replace(old_string, new_string).encode()

    renamed_element = etree.fromstring(string_content, parser)

    return renamed_element
