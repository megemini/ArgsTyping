from __future__ import annotations

import argparse
import re

TYPES: dict[str, str] = {
    "Tensor": "paddle.Tensor",
    "string": "str",
    "function": "Callable[Any, Any]",
    "list": "list[Any]",
    "tuple": "tuple[Any, ...]",
    "dict": "dict[str, Any]",
}
ANNOS: dict[str, str] = {
    "axis": "int | list[int] | tuple[int, ...]",
}
RTNS: dict[str, str] = {
    "self": "Self",
    "Self": "Self",
    "None": "None",
    "none": "None",
    "out": "paddle.Tensor",
}


pat_func_with_doc = re.compile(
    r"""
    [ ]*def[ ]                                              # func begins
        (?P<name>[^(]*?)                                    # name
            \((?P<sig>[^\)]*?)\)                            # signature
            (?P<rtn_anno>[ ]?->[^:]*?)?                     # return annotation
        (?P<sig_end>:)\n      
        [ ]*r?                                              # doc begins
        (?P<quote>[\"\']{3})\n?.*?
            (?P<args>[ ]*((Arg|Parameter)(s)?\:)\n
                (?P<args_content>.*?))?                     # Args
            (?P<rtns>[ ]*((Return|Yield)(s)?\:)\n
                (?P<rtns_content>.*?))?                     # Returns
        (?P=quote)\n
    """,
    re.M | re.S | re.X,
)

pat_args = re.compile(r"[ ]*(?P<name>.*?)\((?P<type>.*?)\)[ ]?:", re.M)
pat_rtn = re.compile(
    r"[ ]*(?P<multi>list of )?(?P<rtn>.*?)(?P<plural>s)?[,:\.\(](?P<desc>.*)"
)


def strip_type(anno: str) -> str:
    return anno.strip().strip("()")


def parse_args(doc: str) -> dict[str, str]:
    args = {}
    if doc:
        for m in pat_args.finditer(doc):
            name = m.group("name")
            types = m.group("type")
            types = [
                TYPES.get(t.strip(), t.strip())
                for t in types.split(",")
                if t.lower().strip() != "optional"
            ]
            args[name.strip()] = " | ".join(types)

    return args


def parse_rtn(doc: str) -> str:
    if doc:
        rtn = doc.splitlines()[0].strip()

        if rtn.strip() in RTNS:
            return RTNS[rtn.strip()]

        match = pat_rtn.match(rtn)
        if match:
            desc = match.group("desc")
            if desc:
                for _rtn in RTNS:
                    if _rtn in desc:
                        return RTNS[_rtn]

            rtn = match.group("rtn").strip().strip("`")
            if match.group("multi") or match.group("plural"):
                return f"list[{TYPES.get(rtn, RTNS.get(rtn, strip_type(rtn)))}]"
            return TYPES.get(rtn, RTNS.get(rtn, strip_type(rtn)))

    return ""


def format_anno(anno: str) -> str:
    return " | ".join(TYPES.get(a.strip(), strip_type(a)) for a in anno.split("|"))


def anno_sig(doc: str, args: dict[str, str]) -> str:
    if not args:
        return doc

    sig = ""
    for arg in doc.split(","):
        arg = arg.strip()

        # parse with default value, `x=None`
        arg = arg.split("=")
        if len(arg) == 1:
            _arg = arg[0].strip()
            value = ""
        elif len(arg) == 2:
            _arg = arg[0].strip()
            value = arg[1].strip()
        else:
            raise
        arg = _arg

        # parse with anno, `x:str=None`
        arg = arg.split(":")
        if len(arg) == 1:
            _arg = arg[0].strip()
            anno = args.get(_arg)
        elif len(arg) == 2:
            _arg = arg[0].strip()
            anno = arg[1].strip()
        else:
            raise
        arg = _arg

        anno = ANNOS.get(arg, anno)

        sig += arg
        if anno:
            anno = format_anno(anno)
            sig += (
                ": "
                + anno
                + (" | None" if value.strip() == "None" and "None" not in anno else "")
            )

        if value:
            sig += (" = " if anno else "=") + value

        sig += ", "

    return sig.strip(", ")


def anno_rtn_anno(doc: str, rtn: str) -> str:
    if doc is not None and doc.strip() and "->" in doc:
        return " " + doc.strip()

    if rtn is not None and rtn.strip():
        return " -> " + rtn

    return ""


def annotate(input_file: str, output_file: str) -> None:
    codes = ""
    with open(input_file) as f:
        codes = "".join(f.readlines())

    _codes = []
    for i, m in enumerate(pat_func_with_doc.finditer(codes)):
        i_0, _ = m.span()
        args = parse_args(m.group("args_content"))
        rtn = parse_rtn(m.group("rtns_content"))

        sig = anno_sig(m.group("sig"), args)
        sig_0, sig_1 = m.span("sig")

        rtn_anno = m.group("rtn_anno")

        if i == 0:
            _codes.extend([codes[0 if i == 0 else i_0 : sig_0], sig])
        else:
            _codes.extend([codes[i_1:i_0], codes[i_0:sig_0], sig])

        _, i_1 = m.span()

        if rtn_anno is None:
            rtn_anno = anno_rtn_anno(rtn_anno, rtn)
            sig_end_0, sig_end_1 = m.span("sig_end")
            _codes.extend([codes[sig_1:sig_end_0], rtn_anno, ":", codes[sig_end_1:i_1]])

        else:
            rtn_anno = anno_rtn_anno(rtn_anno, rtn)
            rtn_0, rtn_1 = m.span("rtn_anno")
            _codes.extend([codes[sig_1:rtn_0], rtn_anno, codes[rtn_1:i_1]])

    _codes.append(codes[i_1:])

    if output_file:
        with open(output_file, "w") as f:
            f.write("".join(_codes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        default="",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file or input_file

    annotate(input_file, output_file)
