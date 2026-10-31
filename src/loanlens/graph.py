from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict,total=False):
  postcode:str
  region:str


def normalise(state:State) -> dict:
  return {"postcode":state["postcode"].upper().strip()}

def find_region(state:State) -> dict:
  return {"region":"London" if state["postcode"].startswith("SW") else "Others"}

builder = StateGraph(State)
builder.add_node(normalise)
builder.add_node(find_region)

builder.add_edge(START,"normalise")
builder.add_edge("normalise","find_region")
builder.add_edge("find_region",END)

graph=builder.compile()

print(graph.invoke({"postcode":" sw1a 1aa "}))

