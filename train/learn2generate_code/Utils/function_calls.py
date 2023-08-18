import pubchempy as pcp
import re


def compensate_for_loss(loss="0%", current_value=0) -> float:
    amount = re.search(r"(\d+\.*\d+)", loss).group()
    if "%" in loss:
        return current_value * (100 / (float(amount) + 100))
    else:
        return current_value + float(amount)


def to_mg(passed_product):
    product_described = float(re.search("(\d+\.?\d*)", passed_product).group())
    if 'mg' in product_described or 'mL' in product_described or 'ml' in product_described:
        proudct_in_mg = product_described
    elif 'g' in product_described or 'gr' in product_described or 'L' in product_described:
        proudct_in_mg = product_described * 1000
    else:
        assert 'kg' in product_described
        proudct_in_mg = product_described * 1000000
    return proudct_in_mg


def to_stu(passed_product):
    product_described = float(re.search("(\d+\.?\d*)", passed_product).group())
    if 'mg' in passed_product or 'mL' in passed_product or 'ml' in passed_product:
        product_in_gr = product_described / 1000
    elif ' g' in passed_product or 'gr' in passed_product or 'L' in passed_product:
        product_in_gr = product_described
    else:
        if 'kg' not in passed_product:
            print("could not find a unit to convert to gram\t",product_described)
            print()
            raise Exception
        product_in_gr = product_described * 1000
    return product_in_gr


def compensate_for_loss(loss="0%", current_value=0) -> float:
    amount = re.search(r"(\d+\.*\d+)", loss).group()
    if "%" in loss:
        return current_value * (100 / (float(amount) + 100))
    else:
        return current_value + float(amount)


def to_minute(ss):
    return 1
