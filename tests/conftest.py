from jaxtyping import install_import_hook

# import gpjax within import hook to apply beartype everywhere, before running tests
# with install_import_hook("gpjax", "beartype.beartype"):
#    import gpjax  # noqa: F401
