__all__ = ["serialize", "get_serialize_func"]


def val_transform(val, valType="tabular"):
    if type(val) == str and len(val.split()) > 100:
        val = val.split(".")[0]
    elif (
        ((type(val) == float or type(val) == int) and val == 0)
        or (len(str(str)) == 0)
        or (str(val) == "nan")
    ):
        if valType == "tabular":
            val = "NULL"
        else:
            val = "missing"
    return str(val)


def single_entry_serialize(cols, vals, rm_null=False):
    data = []
    for col, val in zip(cols, vals):
        val = val_transform(val)
        if rm_null and (val == "NULL" or val == "missing"):
            continue
        data.append(f"{str(col)}: {str(val)}")
    return ", ".join(data)


def single_entry_serialize_NL(cols, vals):
    """Natural language version"""
    data = []
    for col, val in zip(cols, vals):
        val = val_transform(val, valType="NL")
        data.append(f"its {col} is {val}")
    return ", ".join(data)


def serialize_raw(entry_type, cols, valsA, valsB, s_type="s1"):
    """
    s1: Entry A: "attr: val". Entry B: "attr: val"
    s2: Entry: Entry A, Entry B; attr: val1, val2
    s3: For Entry A, its attr is val.
    s4: For Entry A and Entry B, their attr are val1, val2.
    """

    if s_type == "s1":
        dataA = single_entry_serialize(cols, valsA)
        dataB = single_entry_serialize(cols, valsB)
        data = f'{entry_type} A is "{dataA}". {entry_type} B is "{dataB}". '
    elif s_type == "s2":
        dataA = ", ".join([val_transform(valA) for valA in valsA])
        dataB = ", ".join([val_transform(valB) for valB in valsB])
        data = f'{entry_type} A is "{dataA}". {entry_type} B is "{dataB}". '
    elif s_type == "s3":
        dataA = single_entry_serialize_NL(cols, valsA)
        dataB = single_entry_serialize_NL(cols, valsB)
        data = f"For {entry_type} A, {dataA}. For {entry_type} B, {dataB}. "
    elif s_type == "s4":
        dataList = [f"For {entry_type} A and {entry_type} B"]
        for col, valA, valB in zip(cols, valsA, valsB):
            valA = val_transform(valA, valType="NL")
            valB = val_transform(valB, valType="NL")
            dataList.append(f"their {col} are {valA} and {valB}")
        data = ", ".join(dataList)
        data = data + ". "
    elif s_type == "s5":
        dataList = [f"Entry\t{entry_type} A\t{entry_type} B"]
        for col, valA, valB in zip(cols, valsA, valsB):
            valA = val_transform(valA)
            valB = val_transform(valB)
            dataList.append(f"{col}\t{valA}\t{valB}")
        data = "\n".join(dataList)
    elif s_type == "s6":
        valsA = "\t".join([f"{entry_type} A"] + [val_transform(v) for v in valsA])
        valsB = "\t".join([f"{entry_type} B"] + [val_transform(v) for v in valsB])
        cols = "\t".join(["Entry"] + cols)
        data = f"{cols}\n{valsA}\n{valsB}"
    elif s_type == "s7":
        dataA = single_entry_serialize(cols, valsA, rm_null=True)
        dataB = single_entry_serialize(cols, valsB, rm_null=True)
        data = f'{entry_type} A is "{dataA}". {entry_type} B is "{dataB}". '
    elif s_type == "s8":
        dataA, dataB = [], []
        for valA in valsA:
            valA = val_transform(valA)
            if valA != "NULL" and valA != "missing":
                dataA.append(valA)
        for valB in valsB:
            valB = val_transform(valB)
            if valB != "NULL" and valB != "missing":
                dataB.append(valB)
        dataA = ", ".join(dataA)
        dataB = ", ".join(dataB)
        data = f'{entry_type} A is "{dataA}". {entry_type} B is "{dataB}". '
    elif s_type == "s9":
        dataA, dataB = [], []
        for col, valA, valB in zip(cols, valsA, valsB):
            valA = val_transform(valA)
            valB = val_transform(valB)
            if (
                valA != "NULL"
                and valA != "missing"
                and valB != "NULL"
                and valB != "missing"
            ):
                dataA.append(f"{str(col)}: {str(valA)}")
                dataB.append(f"{str(col)}: {str(valB)}")
        dataA = ", ".join(dataA)
        dataB = ", ".join(dataB)
        data = f'{entry_type} A is "{dataA}". {entry_type} B is "{dataB}". '
    elif s_type == "s10":
        dataA, dataB = [], []
        for col, valA, valB in zip(cols, valsA, valsB):
            valA = val_transform(valA)
            valB = val_transform(valB)
            if (
                valA != "NULL"
                and valA != "missing"
                and valB != "NULL"
                and valB != "missing"
            ):
                dataA.append(valA)
                dataB.append(valB)
        dataA = ", ".join(dataA)
        dataB = ", ".join(dataB)
        data = f'{entry_type} A is "{dataA}". {entry_type} B is "{dataB}". '
    elif s_type == "s11":
        valsA = "\t".join([f"{entry_type} A"] + [val_transform(v) for v in valsA])
        valsB = "\t".join([f"{entry_type} B"] + [val_transform(v) for v in valsB])
        data = f"{valsA}\n{valsB}"
    return data


def serialize(entry_type, entry_pair, s_type="s1"):
    return serialize_raw(
        entry_type, entry_pair.cols, entry_pair.valsA, entry_pair.valsB, s_type
    )


def get_serialize_func(entry_type, s_type):
    return lambda entry_pair: serialize(entry_type, entry_pair, s_type)
