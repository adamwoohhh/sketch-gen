def test_package_exports_public_python_api():
    from sketch_gen import RenderOptions, RenderResult, render_sketch_gif

    assert RenderOptions.__name__ == "RenderOptions"
    assert RenderResult.__name__ == "RenderResult"
    assert callable(render_sketch_gif)
