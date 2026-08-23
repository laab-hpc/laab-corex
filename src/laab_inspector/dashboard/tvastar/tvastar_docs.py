# Tvastar
#
# Copyright (c) 2026 Aravind Sankaran
# Developed in Aachen, Germany.
#
# Licensed under the BSD 3-Clause License.
# See LICENSE file in the project root for full license information.

from flask import Blueprint, render_template
from . import config
from .actions import examples 

from pathlib import Path

name = Path(__file__).stem


bp = Blueprint(
    name,
    __name__,
    template_folder=f"templates/", 
    static_folder=f"static/",
    static_url_path=f"/static/{name}/",  
)


@bp.route('/')
def index():
    return render_template(f'{name}/index.html')

@bp.route('/cards/')
def cards():
    return render_template(f'{name}/cards.html')

@bp.route('/tables/')
def tables():
    col_schema, pk, df = examples.get_simple_df()
    return render_template(f'{name}/tables.html',
                           col_schema=col_schema,
                           pk=pk,
                           records=df.to_dict(orient="records"))
    
@bp.route('/pr_boxplots/')
def pr_boxplots():
    bp_data, ranks = examples.get_ranked_bp_data()
    return render_template(f'{name}/pr_boxplots.html',
                           bp_data=bp_data,
                           ranks=ranks)
    
@bp.route('/dfg/')
def dfg():
    dfg_svg, node_info = examples.get_dfg_data(config.tvst_eg_data)
    return render_template(f'{name}/dfg.html',
                           dfg_svg=dfg_svg,
                           node_info=node_info)
    
@bp.route('/modals/')
def modals():
    return render_template(f'{name}/modals.html')