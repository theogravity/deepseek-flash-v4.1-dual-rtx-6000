#!/usr/bin/env python3
"""Make vLLM's --api-key check also accept the Anthropic-style `x-api-key` header.

vLLM's AuthenticationMiddleware only accepts `Authorization: Bearer <key>`. LiteLLM's `anthropic/` provider (required for
thinking blocks, see docs/TUNING.md) sends the configured api_key as `x-api-key`, so a protected vLLM port answers
401 to LiteLLM. This patch accepts either header (still SHA-256 + constant-time compare; missing/wrong keys are rejected).

    python3 -I apply_patch.py <stock authenticate.py> <out dir>
Stock file: docker run --rm --entrypoint cat IMAGE /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/middleware/authenticate.py
"""
import sys
from pathlib import Path

OLD = '''    def verify_token(self, headers: Headers) -> bool:
        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()

        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match
'''
NEW = '''    def verify_token(self, headers: Headers) -> bool:
        # Patched (patches/api-auth): also accept the Anthropic-style `x-api-key` header, which LiteLLM's
        # anthropic/ provider sends, in addition to `Authorization: Bearer <key>`.
        candidates = []
        authorization_header_value = headers.get("Authorization")
        if authorization_header_value:
            scheme, _, param = authorization_header_value.partition(" ")
            if scheme.lower() == "bearer":
                candidates.append(param)
        x_api_key = headers.get("x-api-key")
        if x_api_key:
            candidates.append(x_api_key)

        token_match = False
        for candidate in candidates:
            param_hash = hashlib.sha256(candidate.encode("utf-8")).digest()
            for token_hash in self.api_tokens:
                token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match
'''

def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1]).read_text()
    if src.count(OLD) != 1:
        raise SystemExit(f"apply_patch: verify_token anchor found {src.count(OLD)} times (expected 1); the stock file changed")
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    (out / "authenticate.py").write_text(src.replace(OLD, NEW))
    print(f"wrote {out}/authenticate.py")

if __name__ == "__main__":
    main()
