import contextlib
import csv
import json
import os
import pandas
import uuid
import IPython.display

from io import StringIO

_render_dict_counter = 0


def list_to_dict(a_list, delimiter):
    the_dict = {}
    for entry in a_list:
        path = entry.split(delimiter)
        value_holder = the_dict
        for key in path[:-1]:
            value_holder = value_holder.setdefault(key, {})
        value_holder[path[-1]] = 0
    return the_dict


def render_dict(a_dict):
    global _render_dict_counter
    my_uuid = "render_dict_" + str(_render_dict_counter)  # str(uuid.uuid4())
    _render_dict_counter += 1
    IPython.display.display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(my_uuid), raw=True)
    # this is relative to the notebooks directory:
    IPython.display.display_javascript("""
    require(["./renderjson.js"], function() {
      document.getElementById('%s').appendChild(renderjson(%s))
    });
    """ % (my_uuid, json.dumps(a_dict, sort_keys=True)), raw=True)


def list_as_table(a_list, col_width=20, cols=5):
    cell_format = "|{:" + str(col_width) + "." + str(col_width) + "}|"
    return ((cell_format*cols + "\n")*(len(a_list)//cols) + cell_format * (len(a_list) % cols)).format(*a_list)


def dicts_to_df(dicts):
    assert len(dicts) > 0
    csv_buffer = StringIO()
    dict_writer = csv.DictWriter(csv_buffer, dicts[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(dicts)
    csv_buffer.seek(0)
    return pandas.read_csv(csv_buffer)


def format_feature_df(df):
    df.set_index(["Type", "Graph", "Model"], inplace=True, drop=False, verify_integrity=True)
    df.sort_values("Nodes", inplace=True)
    df.sort_index(kind="mergesort", inplace=True)
    df.columns.name = "Feature"


@contextlib.contextmanager
def use_paper_data():
    old_environ = os.environ.copy()
    os.environ["DATA_PATH"] = (
        os.path.dirname(os.path.realpath(__file__)) +
        "/../../data-paper/"
    )
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
