"""Pseudonymisation demo: the model never sees the applicant's name, yet the tools still work.

Flow:  name --HMAC--> token  ·  model only ever sees the token  ·  tool resolves token -> name
       in OUR code  ·  final memo is de-tokenised for the underwriter, in OUR code.
Uses a scripted fake model, so it's offline and free.

Run:  uv run python scripts/pseudonymisation_demo.py
"""

import hashlib
import hmac
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from conftest import ScriptedChatModel
from loanlens.tools import lookup_credit_file

# Keyed hash (HMAC). A PLAIN hash of a name can be reversed by hashing a list of common names.
# In production this secret lives in Azure Key Vault, never in code.
SECRET = b"from-key-vault-never-in-code"

class NameVault:
    def __init__(self): self._real = {}
    def tokenise(self, name):
        token = "APPLICANT_" + hmac.new(SECRET, name.lower().encode(), hashlib.sha256).hexdigest()[:8]
        self._real[token] = name
        return token
    def resolve(self, token): return self._real.get(token, token)
    def detokenise(self, text):
        for t, n in self._real.items(): text = text.replace(t, n)
        return text

def make_tools(vault):
    @tool
    def lookup_credit_file_safe(applicant_name: str) -> dict:
        """Look up the applicant's credit file. Pass the applicant token exactly as given."""
        return lookup_credit_file.invoke({"applicant_name": vault.resolve(applicant_name)})
    return [lookup_credit_file_safe]

vault = NameVault()
tok = vault.tokenise("Sam Jones")
print("token:", tok, "| same again:", vault.tokenise("Sam Jones") == tok)
model = ScriptedChatModel(script=[AIMessage(content="", tool_calls=[{"name":"lookup_credit_file_safe","args":{"applicant_name":tok},"id":"c1"}]), AIMessage(f"REFER TO SENIOR UNDERWRITER\n- {tok} has 2 missed payments")], calls=[], bound_tools=[])
out = create_agent(model, make_tools(vault)).invoke({"messages":[("user", f"Investigate applicant {tok}")]})
seen = " ".join(str(m.content) for call in model.calls for m in call)
print("model ever saw 'Sam Jones'?", "Sam Jones" in seen)
print("tool result:", [m.content[:60] for m in out["messages"] if m.type=="tool"])
print("for the underwriter:", vault.detokenise(out["messages"][-1].content))
