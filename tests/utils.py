import base64
import io
import pytest_html


def svg_to_str(svg):
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.as_svg().encode('utf-8')).decode('utf-8')}"


def mpl_svg_to_str(svg):
    return f"data:image/svg+xml;base64,{base64.b64encode(svg).decode('utf-8')}"


def mpl_fig_to_report(fig, extras):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    byte_data = buf.getvalue()
    extras.append(pytest_html.extras.svg(mpl_svg_to_str(byte_data)))
