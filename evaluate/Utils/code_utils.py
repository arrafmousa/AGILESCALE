import re
import pubchempy as pcp


def formulate_code(entry):
    code = ""
    code += 'import pubchempy as pcp\n'
    code += 'from Utils import function_calls'
    code += '\n'
    code += "def generated_code():\n"
    for code_line in entry.split("[EOL]"):
        sep_var = re.findall("(.*?) _ (.*?)", code_line)
        for idx,sep in enumerate(sep_var):
            changed = sep[0]+"_"+sep[1]
            code_line = code_line.replace(sep[0]+" _ "+sep[1], changed)
        sep_var = re.findall("(.*?). (.*?)", code_line)
        for idx,sep in enumerate(sep_var):
            changed = sep[0]+"."+sep[1]
            code_line = code_line.replace(sep[0]+". "+sep[1], changed)
        if code_line.startswith(" "):
            code_line = code_line[1:]

        if "to_gr" in code_line:
            if "to_gr(\'" not in code_line.replace(" ", "") and "to_gr(\"" not in code_line.replace(" ", "") and \
                    'reactor[1]' not in code_line.replace(" ", "") and \
                    'component[1]' not in code_line.replace(" ", "") and \
                    'have[1]' not in code_line.replace(" ", ""):
                code_line = code_line.replace("to_gr(", "to_gr('")
                code_line = code_line.replace(")", "')")
        code_line = code_line.replace(" (, 'name')[0]", "\",'name')[0]")
        code_line = code_line.replace(",, 'name')[0]", "\",'name')[0]")
        if "pcp.get_compounds(" in code_line:
            if "pcp.get_compounds(\"" not in code_line.replace(" ", ""):
                code_line = code_line.replace("pcp.get_compounds(", "pcp.get_compounds(\"")
            argument = re.search("\"(.*)\"", code_line).group().replace("\"", "")
            args = argument.split(" ")
            args.reverse()
            for arg in args:
                try:
                    pcp.get_compounds(arg, "name")
                    code_line = code_line.replace(argument, arg)
                    code_line = code_line.replace(argument, arg)
                    break
                except:
                    pass
        code_line = code_line.replace("needed_reactors_for_100g_product", "have_components")
        code_line = code_line.replace("[EOL]", "\n")
        code_line = code_line.replace("[TAB]", "\t")

        if len(code_line) > 0 and code_line[0] == "▁":
            code_line = code_line[1:]
        code_line = code_line.replace("▁", " ")
        code += "\t"
        code += code_line
        code += '\n'
    code += '\treturn answer\n'
    code += "ret_vale = generated_code()"
    return code

# formulate_code("desired_product = to_gr( get 777.68 mg of)[EOL]product_described = to_gr( obtain 660 mg of a redd)[EOL]described_component = to_gr( ' 10 ml ')[EOL]needed_reactor = desired_product / product_described * described_component[EOL]reactor_molar_weight = pcp.get_compounds(  and dimethylformamide (, 'name')[0].exact_mass[EOL]return ( needed_reactor / float( reactor_molar_weight ) )[EOL][EOL]")
