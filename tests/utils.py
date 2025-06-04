import base64


def svg_to_str(svg):
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.as_svg().encode('utf-8')).decode('utf-8')}"
