"""MatSci Agent backend package."""

import os

__all__ = ["__version__"]
__version__ = "0.1.0"

# Verify TLS against the operating system's certificate store instead of certifi's bundle. On
# networks that inspect HTTPS (corporate proxies, antivirus), the inspecting CA is installed in the
# OS store only, so Materials Project and arXiv calls otherwise fail with CERTIFICATE_VERIFY_FAILED.
# Verification stays on; only the trust source changes. Opt out with USE_SYSTEM_CERTS=false.
if os.getenv("USE_SYSTEM_CERTS", "true").lower() not in ("0", "false", "no"):
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:  # optional: fall back to the default certificate bundle
        pass
