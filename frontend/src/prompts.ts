export interface PromptTemplate {
  id: string;
  label: string;
  hint: string;
  template: string;
}

export const PROMPT_TEMPLATES: PromptTemplate[] = [
  {
    id: "session-note",
    label: "Draft session note (SOAP)",
    hint: "Structured ABA session note",
    template: `Draft a session note in SOAP format based on the following. Use person-first language. Flag anything that needs clarification before finalizing.

**Client (anonymized):** [age, diagnosis, relevant history]
**Session:** [date, duration, setting]
**Goals targeted:**
  -
  -

**Observations (ABC):**
- Antecedent:
- Behavior:
- Consequence:

**Data summary:** [e.g., % independent trials, frequency, duration]

**Parent/caregiver interactions:**

**Next steps:**
`,
  },
  {
    id: "translate-parents",
    label: "Translate for parents",
    hint: "Clinical → parent-friendly language",
    template: `Rewrite the following clinical text for a parent. Target a 6th-grade reading level. Keep clinical accuracy, replace jargon, and be encouraging without overpromising. Preserve specific behaviors and data.

**Clinical text to translate:**
`,
  },
  {
    id: "review-treatment-plan",
    label: "Review treatment plan",
    hint: "Audit goals, procedures, data collection, progress",
    template: `Review the following ABA treatment plan. Flag anything that's missing, underspecified, or doesn't align.

Check for:
- Assessment summary tied to specific goals
- Operationally defined, measurable goals (with mastery criteria)
- Procedures that actually target each goal
- Data collection methods (what, how often, who)
- Parent/caregiver training plan
- Generalization and maintenance plan
- Reinforcement strategy appropriate to client
- Safety / scope-of-practice boundaries
- Progress toward prior goals (if this is a continuation)
- Anything that conflicts with BACB ethics code

Give your review as a numbered list of issues with severity (minor / needs fix / blocker) and a concrete suggestion for each.

**Treatment plan:**
`,
  },
  {
    id: "review-bip",
    label: "Review BIP for gaps",
    hint: "Audit a Behavior Intervention Plan",
    template: `Review the following Behavior Intervention Plan. Check for and flag anything missing or underspecified:

1. Operational definition of the target behavior
2. Hypothesized function(s)
3. Antecedent strategies
4. Teaching replacement behaviors (including the functional-equivalent skill)
5. Consequence strategies (including what to do for escalations)
6. Fading/generalization plan
7. Crisis plan
8. Data collection methods and decision rules
9. Safety considerations and scope-of-practice boundaries

**BIP:**
`,
  },
  {
    id: "analyze-abc",
    label: "Analyze ABC data",
    hint: "Find patterns and function hypothesis",
    template: `Analyze the following ABC data. In your response, give:
1. Most likely function(s) of the behavior, with evidence
2. Patterns (time of day, setting, activity, people, antecedent conditions)
3. Antecedent-based intervention ideas matched to the hypothesized function
4. Candidate replacement behaviors that serve the same function
5. Any data-quality issues or missing information that would change the analysis

**Client profile (anonymized):**

**ABC data:**
`,
  },
  {
    id: "progress-report",
    label: "Draft progress report",
    hint: "Quarterly-style report",
    template: `Draft a progress report. Structure:
- Executive summary (3-4 sentences)
- Goal-by-goal progress: baseline → current (% change or levels of prompting)
- Notable observations (behavioral, environmental, caregiver)
- Proposed goal updates (add/modify/retire)
- Parent/caregiver collaboration

Use person-first language. Be specific where data is specific; avoid vague claims.

**Client (anonymized):**
**Reporting period:**

**Goals and data:**
1.
2.
3.
`,
  },
  {
    id: "parent-training",
    label: "Parent training plan",
    hint: "Home-program step-by-step",
    template: `Create a parent training plan for the target below. Include:
- Rationale (why this skill, why now)
- Step-by-step procedure parents follow at home
- What to do when it goes right (reinforcement)
- What to do when it goes wrong (correction/redirection — scope of parent role)
- Data collection the parent can realistically do (simple, under 30 seconds)
- Troubleshooting: 3 most common pitfalls
- Criteria to move to next phase

**Target skill/behavior:**
**Client baseline at home:**
**Parent's stated goal or concern:**
`,
  },
  {
    id: "prior-auth",
    label: "Prior authorization letter",
    hint: "Medical-necessity letter to insurance",
    template: `Draft a prior authorization letter to an insurance payer supporting continued ABA services.

Include:
- Clear statement of requested hours per week and duration
- Medical necessity rationale, linked to assessment data
- Treatment goals, stated measurably
- Evidence supporting ABA for this diagnosis (BACB guidelines, peer-reviewed literature — only cite what's real)
- Projected outcomes and criteria for reducing intensity

Tone: professional, concise, data-driven.

**Payer:**
**Client (first name + DOB only):**
**Diagnosis + ICD-10:**
**Assessment snapshot (tool + key scores/ages):**
**Hours requested per week and rationale:**
**Recent clinical progress:**
`,
  },
  {
    id: "appeal-letter",
    label: "Appeal letter (denied claim)",
    hint: "Responds to a payer denial",
    template: `Draft an appeal letter responding to a denied ABA service claim.

Structure:
1. Reference the denial letter (date, claim number, stated reason)
2. Counter the denial rationale directly, pointing to specific errors or missing context
3. Reiterate medical necessity with fresh evidence
4. Cite relevant BACB guidelines, state mandate, or peer-reviewed research where it applies (only cite what's real)
5. Request specific next steps (reconsideration, peer-to-peer review, etc.)

Tone: firm, respectful, evidence-first.

**Denial reason (quoted):**
**Client clinical picture (anonymized):**
**Treatment history + most recent data:**
**Relevant state mandate or plan language if known:**
`,
  },
  {
    id: "interpret-assessment",
    label: "Interpret assessment results",
    hint: "VB-MAPP / ABLLS / Vineland narrative",
    template: `Interpret the following assessment results for a treatment plan narrative.

In your response:
- Summarize strengths and gaps by domain
- Compare to age-typical expectations where relevant
- Identify 3-5 priority goal domains with rationale
- Keep clinical accuracy while using language accessible to parents
- Flag anything the raw scores suggest but don't confirm (e.g., need for follow-up assessment)

**Assessment tool:**
**Client age:**
**Key scores/milestones:**
`,
  },
  {
    id: "generalization",
    label: "Generalization plan",
    hint: "Plan to generalize a mastered skill",
    template: `Help me plan generalization and maintenance for a mastered skill.

Cover:
- Stimulus generalization: new people, settings, materials (give specific examples)
- Response generalization: variations of the response that should be reinforced
- Maintenance schedule (probes at what frequency, for how long)
- Probe data collection method
- Criteria to conclude generalization has occurred
- Common failure modes and how to catch them early

**Mastered skill:**
**Mastery criteria that was met:**
**Current setting + people present during acquisition:**
**Desired generalization targets:**
`,
  },
];
