# LoanLens — design sketch (Lesson 0)

Answer from memory, in your own words. Short is fine. Interviewers ask exactly these.

## 1. Which ONE part of LoanLens should be a true agent, and why that part?
The part which checks for the risk and compliance after the policy rules are checked. Because this is the part where LLM might have to decide which step/tool to call in a loop;
>

## 2. Name two parts that must be plain code, never an LLM. What would go wrong if an LLM did them?
The part which is checking against the policy rules. If an LLM does this there can be chances that it might miss a rule.
The other part is Redact PII - this should be the code one to remove PII and LLM will miss removing some details and we will end up sending the personal details to llm.

>

## 3. What is the agent's stopping condition? What happens if it doesn't have one?
The natural stop would be when the model stops asking for additional tools and the bard stop would be after a maximum number of loop steps. An agent can't run forever as it will increase the cost,latency and hit rate limits.

>

## 4. Where must a human approve, and why there — not earlier, not later?
Human must approve after the LLM does the evaluation against the risk and compliance after seeing agent's response. If done earlier then he has to go through the risk and compliance himself. 

>
