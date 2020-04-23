from IPython.core.display import display, HTML
from analysis import *

def _show(fp):
    display(HTML(open(fp).read()))
    
def guess():
    _show("Kontaktok_számának_becslési_pontossága.html")
    
def gender():
    _show("Kontaktok_megoszlása_nemek_szerint.html")
    
def position():
    _show("Kontaktok_megoszlása_pozíció_szerint.html")
    
def accomm():
    _show("Kontaktok_megoszlása_tartózkodási_hely_szerint.html")
    

def platform():
    _show("Kommunikációs_platformok_használatának_megoszlása.html")
    
    
def topics():
    _show("Témák_népszerűsége.html")

    
def network():
    _show("Jeszkes_kapcsolattartási_hálózat.html")
    
    
def reg_results():
    fit_logit(X)