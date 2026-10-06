import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schema import PromptItem, GroundTruth, PromptDataset
from src.data.validation import validate_dataset


def generate_prompts() -> PromptDataset:
    prompts = []

    # -------------------------------------------------------------
    # 1. INSTRUCTION PROMPTS (20 prompts: p001 - p020)
    # -------------------------------------------------------------
    instructions_data = [
        (
            "p001",
            "That cat chased the mouse across the garden.",
            "Narrate or describe that the specific indicated cat chased the mouse across the garden.",
            [], ["declarative", "neutral"], ["cat chasing mouse across garden"], []
        ),
        (
            "p002",
            "The cat chased the mouse across the garden.",
            "Narrate or describe that a cat chased the mouse across the garden.",
            [], ["declarative", "neutral"], ["cat chasing mouse across garden"], []
        ),
        (
            "p003",
            "Summarize the main findings of the attached quarterly earnings report.",
            "Summarize quarterly earnings report findings.",
            [], ["concise", "informative"], ["quarterly financial report"], []
        ),
        (
            "p004",
            "Translate the following paragraph into idiomatic French.",
            "Translate given paragraph into French.",
            [], ["fluent", "neutral"], ["source text paragraph"], []
        ),
        (
            "p005",
            "Extract all action items and assignees mentioned in the meeting minutes.",
            "Extract action items and assignees from meeting minutes.",
            [], ["objective", "structured"], ["meeting minutes record"], []
        ),
        (
            "p006",
            "Generate five candidate headlines for a blog post about renewable energy advances.",
            "Generate five blog post headlines about renewable energy.",
            [], ["engaging", "creative"], ["renewable energy topic"], []
        ),
        (
            "p007",
            "Explain the difference between supervised and unsupervised machine learning.",
            "Explain difference between supervised and unsupervised learning.",
            [], ["educational", "clear"], ["machine learning paradigms"], []
        ),
        (
            "p008",
            "Convert the CSV table below into valid JSON array format.",
            "Convert CSV formatted data into JSON format.",
            [], ["technical", "precise"], ["input CSV table"], []
        ),
        (
            "p009",
            "Proofread the draft email and correct any grammatical or spelling mistakes.",
            "Proofread draft email and fix grammar and spelling errors.",
            [], ["constructive", "helpful"], ["draft email content"], []
        ),
        (
            "p010",
            "Draft a professional reply declining the job offer with gratitude.",
            "Draft a polite job offer rejection email.",
            [], ["polite", "professional"], ["received job offer"], []
        ),
        (
            "p011",
            "Describe the life cycle of a monarch butterfly from egg to adult.",
            "Describe monarch butterfly biological life cycle stages.",
            [], ["informative", "educational"], ["biological metamorphosis"], []
        ),
        (
            "p012",
            "Refactor this Python function to improve algorithmic time complexity.",
            "Refactor Python code for better time complexity.",
            [], ["technical", "analytical"], ["target Python function"], []
        ),
        (
            "p013",
            "List three pros and three cons of transitioning to a microservices architecture.",
            "List three advantages and three disadvantages of microservices.",
            [], ["balanced", "analytical"], ["software architecture paradigms"], []
        ),
        (
            "p014",
            "Write a brief plot synopsis for a mystery novel set in an isolated lighthouse.",
            "Write mystery novel plot synopsis set in a lighthouse.",
            [], ["atmospheric", "intriguing"], ["isolated lighthouse setting"], []
        ),
        (
            "p015",
            "Identify the logical fallacies present in the provided debate excerpt.",
            "Identify logical fallacies in debate text.",
            [], ["critical", "objective"], ["debate transcript excerpt"], []
        ),
        (
            "p016",
            "Generate unit test cases for an email address validation utility function.",
            "Generate comprehensive unit tests for email validator function.",
            [], ["rigorous", "technical"], ["email validation specification"], []
        ),
        (
            "p017",
            "Define the concept of opportunity cost using a relatable everyday example.",
            "Define opportunity cost with an everyday example.",
            [], ["accessible", "instructional"], ["economic principles"], []
        ),
        (
            "p018",
            "Classify the sentiment of the following customer product reviews as positive, negative, or neutral.",
            "Classify sentiment of customer reviews into three categories.",
            [], ["objective", "analytical"], ["customer product reviews"], []
        ),
        (
            "p019",
            "Formulate three thoughtful questions to ask the interviewer at the end of a software engineering interview.",
            "Formulate three candidate questions for software engineering interviewer.",
            [], ["inquisitive", "professional"], ["job interview conclusion"], []
        ),
        (
            "p020",
            "Summarize Newton's three laws of motion with one practical application for each.",
            "Summarize Newton's laws of motion with practical applications.",
            [], ["educational", "structured"], ["classical mechanics physics"], []
        ),
    ]

    for pid, prompt, intent, constraints, tone, context, steps in instructions_data:
        prompts.append(PromptItem(
            id=pid,
            category="instruction",
            prompt=prompt,
            ground_truth=GroundTruth(
                intent=intent,
                role="assistant",
                constraints=constraints,
                tone=tone,
                context=context,
                steps=steps
            )
        ))

    # -------------------------------------------------------------
    # 2. CONSTRAINT-HEAVY PROMPTS (20 prompts: p021 - p040)
    # -------------------------------------------------------------
    constraints_data = [
        (
            "p021",
            "Summarize the article in exactly five sentences. Do not use technical terminology. Do not mention the author's name.",
            "Summarize the article adhering to strict length, vocabulary, and entity omissions.",
            ["exactly five sentences", "no technical terminology", "do not mention author's name"],
            ["concise", "accessible"], ["article content"], []
        ),
        (
            "p022",
            "Explain quantum computing to a high schooler. Use fewer than 100 words and do not mention qubits or superposition.",
            "Explain quantum computing simply under word count and banned vocabulary constraints.",
            ["under 100 words", "target audience high school student", "do not use word qubits", "do not use word superposition"],
            ["simple", "engaging"], ["quantum computing concepts"], []
        ),
        (
            "p023",
            "Write a product description for running shoes. You must not use the words comfortable, fast, or lightweight.",
            "Write running shoes product copy without banned buzzwords.",
            ["must not use word comfortable", "must not use word fast", "must not use word lightweight"],
            ["persuasive", "commercial"], ["running shoes product"], []
        ),
        (
            "p024",
            "Provide troubleshooting steps for Wi-Fi failure. Output must be strictly valid JSON without any markdown formatting.",
            "Provide Wi-Fi troubleshooting in raw JSON format with no markdown.",
            ["output strictly valid JSON", "no markdown formatting", "no conversational commentary"],
            ["technical", "programmatic"], ["Wi-Fi connectivity issue"], []
        ),
        (
            "p025",
            "Describe the French Revolution in exactly three bullet points. Each bullet must be between 15 and 20 words.",
            "Summarize French Revolution with strict bullet count and exact word length range per bullet.",
            ["exactly three bullet points", "each bullet between 15 and 20 words"],
            ["historical", "factual"], ["French Revolution"], []
        ),
        (
            "p026",
            "Write an apology letter to a client. Never admit legal liability and do not offer monetary refunds.",
            "Draft client apology with strict legal liability exclusions and refund prohibitions.",
            ["never admit legal liability", "do not offer monetary refund"],
            ["apologetic", "cautious", "professional"], ["service disruption incident"], []
        ),
        (
            "p027",
            "Generate an executive summary of the proposal in under 50 words using only passive voice.",
            "Generate executive summary with strict length ceiling and passive voice constraint.",
            ["under 50 words", "only passive voice syntax"],
            ["formal", "executive"], ["business proposal text"], []
        ),
        (
            "p028",
            "List five dinner recipes that are completely gluten-free, dairy-free, and nut-free with at least 25g protein per serving.",
            "Recommend five dinner recipes meeting multiple dietary exclusions and a minimum protein threshold.",
            ["exactly five recipes", "gluten-free", "dairy-free", "nut-free", "at least 25g protein per serving"],
            ["practical", "nutritional"], ["dinner meal planning"], []
        ),
        (
            "p029",
            "Write a haiku about autumn. The second line must contain the color crimson.",
            "Compose autumn haiku with specific color word in line two.",
            ["5-7-5 haiku structure", "second line must contain word crimson"],
            ["poetic", "evocative"], ["autumn season theme"], []
        ),
        (
            "p030",
            "Explain photosynthesis. Do not use the letter 'e' anywhere in your response.",
            "Explain photosynthesis under a lipogram constraint forbidding the letter 'e'.",
            ["lipogram forbidding letter 'e'", "accurate explanation of photosynthesis"],
            ["educational", "constrained"], ["photosynthesis process"], []
        ),
        (
            "p031",
            "Review this resume. Limit feedback to three major weaknesses and do not suggest cosmetic changes.",
            "Review resume identifying weaknesses while excluding formatting/cosmetic critique.",
            ["limit to exactly three major weaknesses", "do not suggest formatting or cosmetic changes"],
            ["constructive", "direct"], ["candidate resume document"], []
        ),
        (
            "p032",
            "Write an SQL query to retrieve active users. Do not use subqueries or the JOIN operator.",
            "Write SQL query retrieving active users avoiding subqueries and joins.",
            ["no subqueries", "no JOIN operator", "retrieve active users"],
            ["technical", "code-only"], ["database schema"], []
        ),
        (
            "p033",
            "Write an incident report for an outage. Under no circumstances should internal server IP addresses be revealed.",
            "Write incident report with strict redaction of internal infrastructure IPs.",
            ["under no circumstances reveal internal server IP addresses", "maintain factual timeline"],
            ["formal", "security-conscious"], ["server outage incident"], []
        ),
        (
            "p034",
            "Draft a motivational speech for high school graduates. Keep the duration under two minutes and avoid cliches like chase your dreams.",
            "Draft high school graduation speech under duration constraint avoiding specific cliches.",
            ["under two minutes speaking duration", "avoid phrase 'chase your dreams'", "avoid common graduation cliches"],
            ["inspiring", "authentic"], ["high school graduation ceremony"], []
        ),
        (
            "p035",
            "Summarize the legal contract terms. Use only plain English readable at an eighth-grade level with no Latin phrases.",
            "Summarize legal contract in eighth-grade reading level without Latin phrases.",
            ["eighth-grade reading level", "no Latin phrases", "plain English only"],
            ["accessible", "clear"], ["legal contract terms"], []
        ),
        (
            "p036",
            "Create a workout plan for a beginner recovering from knee surgery. Prohibit all high-impact jumping and deep squats.",
            "Create beginner workout plan strictly excluding knee-straining movements.",
            ["prohibit high-impact jumping", "prohibit deep squats", "suitable for knee surgery rehabilitation"],
            ["encouraging", "safety-focused"], ["beginner recovering from knee surgery"], []
        ),
        (
            "p037",
            "Write a story opening of 80 words where every sentence begins with the word 'When'.",
            "Write narrative opening constrained by sentence initial word and length limit.",
            ["around 80 words", "every sentence must begin with 'When'"],
            ["narrative", "stylistic"], ["fiction opening"], []
        ),
        (
            "p038",
            "Provide three counterarguments to the policy proposal. Do not attack the character of the proposer.",
            "Provide policy counterarguments restricted to substantive policy rather than ad hominem attacks.",
            ["exactly three counterarguments", "no ad hominem attacks or character critiques"],
            ["analytical", "civil"], ["public policy proposal"], []
        ),
        (
            "p039",
            "Write a product FAQ entry for international shipping. State costs must not be estimated and delivery guarantees cannot be promised.",
            "Write international shipping FAQ with strict prohibitions on cost estimates and delivery guarantees.",
            ["must not estimate shipping costs", "cannot promise guaranteed delivery dates"],
            ["informative", "legally accurate"], ["international shipping policy"], []
        ),
        (
            "p040",
            "Summarize the patient lab results. Use only metric units and do not provide medical diagnosis or prognosis.",
            "Summarize lab results in metric units while strictly avoiding diagnosis or prognosis.",
            ["metric units only", "do not diagnose", "do not offer prognosis"],
            ["objective", "clinical"], ["patient blood test panel"], []
        ),
    ]

    for pid, prompt, intent, constraints, tone, context, steps in constraints_data:
        prompts.append(PromptItem(
            id=pid,
            category="constraint",
            prompt=prompt,
            ground_truth=GroundTruth(
                intent=intent,
                role="assistant",
                constraints=constraints,
                tone=tone,
                context=context,
                steps=steps
            )
        ))

    # -------------------------------------------------------------
    # 3. CONTEXT-HEAVY PROMPTS (20 prompts: p041 - p060)
    # -------------------------------------------------------------
    context_data = [
        (
            "p041",
            "You are responding to a customer who has already contacted support twice regarding a delayed shipment order #9482. Carrier tracking has been stuck in transit for 6 days.",
            "Respond to customer with full acknowledgment of prior support contacts and delayed carrier tracking.",
            [], ["empathetic", "accountable"],
            ["customer contacted support twice previously", "delayed shipment order #9482", "carrier tracking stuck for 6 days"],
            []
        ),
        (
            "p042",
            "The client is a Tier-1 enterprise financial institution with strict HIPAA and SOC2 compliance mandates evaluating cloud migration to AWS Ireland region.",
            "Provide cloud migration recommendations tailored to strict financial and data sovereignty compliance mandates.",
            [], ["consultative", "authoritative"],
            ["client is Tier-1 enterprise financial institution", "subject to HIPAA and SOC2 mandates", "migrating to AWS Ireland region"],
            []
        ),
        (
            "p043",
            "Our company recently laid off 15% of engineering staff following a cancelled Series B funding round. Address remaining employees in an internal memo.",
            "Draft all-hands internal memo addressing workforce reduction after cancelled funding round.",
            [], ["transparent", "reassuring", "sober"],
            ["company laid off 15% engineering staff", "cancelled Series B funding round", "internal memo to retained employees"],
            []
        ),
        (
            "p044",
            "A patient with type 2 diabetes and mild chronic kidney disease is considering starting a ketogenic diet. Evaluate dietary risks given these comorbidities.",
            "Evaluate clinical risks of ketogenic diet accounting for diabetes and kidney disease comorbidities.",
            [], ["clinical", "cautious"],
            ["patient has type 2 diabetes", "patient has chronic kidney disease", "patient considering ketogenic diet"],
            []
        ),
        (
            "p045",
            "We operate a distributed microservices platform experiencing intermittent 504 gateway timeouts on the checkout service during Black Friday peak traffic.",
            "Formulate architectural diagnosis and mitigation plan for Black Friday checkout gateway timeouts.",
            [], ["urgent", "diagnostic"],
            ["distributed microservices architecture", "intermittent 504 timeouts on checkout", "Black Friday peak traffic surge"],
            []
        ),
        (
            "p046",
            "The user is an absolute beginner in programming who just installed Python 3.12 on Windows 11 and encountered a 'python is not recognized' command error.",
            "Guide absolute beginner Windows user through PATH environment variable fix for Python installation.",
            [], ["supportive", "step-by-step"],
            ["user is absolute beginner", "running Python 3.12 on Windows 11", "encountered PATH recognition error"],
            []
        ),
        (
            "p047",
            "The prospective client has a strict budget of $20,000, a hard launch deadline in 6 weeks, and wants a cross-platform mobile app with real-time video chat.",
            "Assess project feasibility balancing fixed budget, compressed deadline, and complex video feature scope.",
            [], ["realistic", "strategic"],
            ["budget capped at $20,000", "hard launch deadline in 6 weeks", "scope demands cross-platform real-time video chat"],
            []
        ),
        (
            "p048",
            "The target demographic is elderly retirees aged 70+ in rural communities with limited broadband access and low smartphone literacy.",
            "Design communication and UX approach tailored to rural elderly demographic with connectivity barriers.",
            [], ["accessible", "respectful"],
            ["target demographic elderly retirees 70+", "rural geography with limited broadband", "low smartphone literacy"],
            []
        ),
        (
            "p049",
            "We are a seed-stage B2B SaaS startup with 4 months of runway remaining and an average sales cycle of 90 days. Advise on growth priorities.",
            "Advise startup founders on survival and sales pipeline priorities under critical runway constraints.",
            [], ["pragmatic", "decisive"],
            ["seed-stage B2B SaaS startup", "4 months runway remaining", "sales cycle is 90 days"],
            []
        ),
        (
            "p050",
            "The dataset exhibits severe class imbalance with positive fraud cases accounting for 0.05% of 10 million transaction records. Recommend modeling strategy.",
            "Recommend fraud detection modeling strategy addressing extreme class imbalance in large-scale transactions.",
            [], ["analytical", "rigorous"],
            ["extreme class imbalance with 0.05% positive cases", "10 million transaction records", "financial fraud classification task"],
            []
        ),
        (
            "p051",
            "The restaurant has suffered a 20% drop in foot traffic following local road construction that will block direct street parking for the next four months.",
            "Formulate marketing and delivery strategy to offset traffic loss from prolonged road construction.",
            [], ["constructive", "resourceful"],
            ["restaurant foot traffic dropped 20%", "road construction blocks street parking", "disruption lasts four months"],
            []
        ),
        (
            "p052",
            "The student scored 98% in math, 95% in physics, but 62% in literature, causing overall GPA to dip below honors eligibility. Draft advisory counseling feedback.",
            "Draft academic counseling advice addressing STEM excellence alongside humanities deficit affecting honors eligibility.",
            [], ["encouraging", "constructive"],
            ["student excels in math (98%) and physics (95%)", "student struggles in literature (62%)", "overall GPA below honors cutoff"],
            []
        ),
        (
            "p053",
            "The API endpoint receives 15,000 requests per second with strict p99 latency SLA of 15ms. The Redis cache hit rate recently fell from 94% to 71%.",
            "Investigate performance degradation and latency SLA risk resulting from cache hit rate drop under high load.",
            [], ["technical", "investigative"],
            ["15,000 requests per second traffic", "15ms p99 latency SLA", "Redis cache hit rate dropped from 94% to 71%"],
            []
        ),
        (
            "p054",
            "The tenant has lived in the apartment for three years with impeccable payment history, but the property manager just served an unexpected 35% rent increase notice.",
            "Draft rent negotiation letter leveraging immaculate tenant payment track record against excessive increase.",
            [], ["firm", "reasonable", "courteous"],
            ["tenant resident for three years with perfect payment history", "received unexpected 35% rent increase", "landlord tenant negotiation"],
            []
        ),
        (
            "p055",
            "The applicant has 7 years of backend Java experience but zero experience with Python or machine learning, applying for an AI Systems Engineer position.",
            "Evaluate resume match and provide transition roadmap for experienced backend engineer entering AI engineering.",
            [], ["objective", "constructive"],
            ["applicant has 7 years Java backend experience", "applicant has zero Python or ML background", "role is AI Systems Engineer"],
            []
        ),
        (
            "p056",
            "The manufacturing facility operates in high-humidity tropical conditions where standard steel bearings corrode within three months instead of expected 18-month lifespan.",
            "Recommend industrial bearing material and lubrication upgrades suited for tropical high-humidity environment.",
            [], ["engineering-focused", "practical"],
            ["tropical high-humidity manufacturing facility", "steel bearings corrode in 3 months vs 18 months standard", "corrosion failure issue"],
            []
        ),
        (
            "p057",
            "The e-commerce store sells perishable gourmet cheeses shipped with dry ice. Summertime ground shipping heat waves are causing delivery spoilage claims to surge 40%.",
            "Propose packaging, courier, and cold-chain logistics adjustments to stop cheese spoilage during summer heatwaves.",
            [], ["logistical", "actionable"],
            ["e-commerce seller of perishable gourmet cheeses", "uses dry ice packaging", "summer heatwaves causing 40% rise in spoilage claims"],
            []
        ),
        (
            "p058",
            "The hospital emergency department triage system is seeing unprecedented patient surge following a multi-vehicle highway collision during severe blizzard conditions.",
            "Establish emergency triage resource allocation protocol during mass casualty incident in severe weather.",
            [], ["decisive", "urgent", "clinical"],
            ["hospital ED facing mass casualty surge", "multi-vehicle highway accident cause", "severe blizzard weather impedes transport"],
            []
        ),
        (
            "p059",
            "The open-source repository has 12,000 GitHub stars but only one maintainer, facing 450 unanswered pull requests and community burnout.",
            "Advise solo maintainer on community governance, triage automation, and sustainability strategy.",
            [], ["supportive", "strategic"],
            ["popular repo with 12k GitHub stars", "single solo maintainer facing burnout", "450 backlog pull requests"],
            []
        ),
        (
            "p060",
            "The user is migrating a legacy monolithic database with 500 million rows from MySQL 5.6 to PostgreSQL 16 with zero scheduled downtime allowed.",
            "Design zero-downtime database replication and cutover architecture from legacy MySQL to Postgres.",
            [], ["architectural", "rigorous"],
            ["legacy MySQL 5.6 database with 500M rows", "migrating to PostgreSQL 16", "zero scheduled downtime constraint"],
            []
        ),
    ]

    for pid, prompt, intent, constraints, tone, context, steps in context_data:
        prompts.append(PromptItem(
            id=pid,
            category="context",
            prompt=prompt,
            ground_truth=GroundTruth(
                intent=intent,
                role="specialist",
                constraints=constraints,
                tone=tone,
                context=context,
                steps=steps
            )
        ))

    # -------------------------------------------------------------
    # 4. TONE/STYLE PROMPTS (20 prompts: p061 - p080)
    # -------------------------------------------------------------
    tone_data = [
        (
            "p061",
            "Explain the service outage to a frustrated customer. Be deeply empathetic, professional, reassuring, and concise.",
            "Explain outage to angry customer with empathy, professionalism, and conciseness.",
            [], ["deeply empathetic", "professional", "reassuring", "concise"],
            ["service outage affecting customer"], []
        ),
        (
            "p062",
            "Draft a cease-and-desist letter to a copyright infringer. Adopt a stern, formal, unambiguous, and authoritative legal tone.",
            "Draft legal cease-and-desist notice with stern authoritative tone.",
            [], ["stern", "formal", "unambiguous", "authoritative"],
            ["copyright infringement infringement notice"], []
        ),
        (
            "p063",
            "Write a product announcement for our breakthrough AR glasses. Sound enthusiastic, visionary, inspiring, and electric.",
            "Write product announcement with passionate and visionary excitement.",
            [], ["enthusiastic", "visionary", "inspiring", "electric"],
            ["breakthrough AR glasses launch"], []
        ),
        (
            "p064",
            "Deliver constructive performance review feedback to an underperforming team member. Maintain a compassionate, encouraging, yet honest tone.",
            "Provide performance critique balancing compassion, honesty, and encouragement.",
            [], ["compassionate", "encouraging", "honest", "constructive"],
            ["performance evaluation meeting"], []
        ),
        (
            "p065",
            "Draft a crisis press release regarding an executive resignation. Ensure the tone is calm, dignified, measured, and forward-looking.",
            "Draft executive departure announcement preserving corporate stability and dignity.",
            [], ["calm", "dignified", "measured", "forward-looking"],
            ["executive sudden resignation crisis"], []
        ),
        (
            "p066",
            "Write a children's bedtime story about a sleepy owl. Use a whimsical, gentle, soothing, and rhythmic voice.",
            "Write bedtime story using whimsical soothing cadence.",
            [], ["whimsical", "gentle", "soothing", "rhythmic"],
            ["bedtime owl story"], []
        ),
        (
            "p067",
            "Decline a collaborative partnership inquiry from a competitor. Remain cordial, diplomatic, firm, and respectful.",
            "Politely decline competitor partnership inquiry with diplomatic firmness.",
            [], ["cordial", "diplomatic", "firm", "respectful"],
            ["competitor partnership proposal"], []
        ),
        (
            "p068",
            "Write a rally speech for volunteers cleaning up a polluted beach. Speak with infectious passion, camaraderie, and urgent optimism.",
            "Deliver volunteer beach cleanup rally speech.",
            [], ["infectious passion", "camaraderie", "urgent optimism"],
            ["community beach cleanup event"], []
        ),
        (
            "p069",
            "Deliver an executive briefing on supply chain risks to the board of directors. Keep the tone sober, analytical, objective, and urgent.",
            "Present supply chain risks to board members objectively and soberly.",
            [], ["sober", "analytical", "objective", "urgent"],
            ["board of directors executive risk briefing"], []
        ),
        (
            "p070",
            "Explain modern slang terms to a grandparent. Sound warm, humorous, patient, and affectionate.",
            "Explain slang to elderly grandparent with humor and patience.",
            [], ["warm", "humorous", "patient", "affectionate"],
            ["generational slang explanation"], []
        ),
        (
            "p071",
            "Write an obituary honoring a dedicated community doctor. Express profound gratitude, reverence, solemnity, and comfort.",
            "Compose physician obituary with reverence and community gratitude.",
            [], ["profound gratitude", "reverent", "solemn", "comforting"],
            ["beloved community doctor memorial"], []
        ),
        (
            "p072",
            "Send a follow-up payment reminder for an invoice that is 60 days overdue. Be polite, firm, uncompromising, and direct.",
            "Send overdue debt collection reminder balancing politeness and uncompromising firmness.",
            [], ["polite", "firm", "uncompromising", "direct"],
            ["invoice overdue by 60 days"], []
        ),
        (
            "p073",
            "Introduce a new office recycling policy. Keep the message playful, engaging, lighthearted, yet persuasive.",
            "Announce office recycling policy in an upbeat fun style.",
            [], ["playful", "engaging", "lighthearted", "persuasive"],
            ["workplace green recycling initiative"], []
        ),
        (
            "p074",
            "Console a friend who did not receive their dream university admission. Be nurturing, comforting, validating, and resilient.",
            "Console rejected applicant friend with nurturing validation and resilience.",
            [], ["nurturing", "comforting", "validating", "resilient"],
            ["university rejection letter"], []
        ),
        (
            "p075",
            "Announce an annual company retreat in Hawaii. Sound celebratory, vibrant, energizing, and inclusive.",
            "Announce corporate Hawaii retreat with infectious celebratory energy.",
            [], ["celebratory", "vibrant", "energizing", "inclusive"],
            ["annual company retreat in Hawaii"], []
        ),
        (
            "p076",
            "Write a restaurant critique of a pretentious fine-dining venue. Employ a sarcastic, witty, sophisticated, and acerbic tone.",
            "Critique pretentious restaurant using biting wit and sophisticated sarcasm.",
            [], ["sarcastic", "witty", "sophisticated", "acerbic"],
            ["high-end fine dining critique"], []
        ),
        (
            "p077",
            "Address citizens regarding an impending hurricane landfall. Maintain an authoritative, urgent, steady, and protective demeanor.",
            "Deliver civil emergency hurricane warning with steady authoritative urgency.",
            [], ["authoritative", "urgent", "steady", "protective"],
            ["imminent category 4 hurricane landfall"], []
        ),
        (
            "p078",
            "Welcome a newly hired software engineer on their first day. Radiate warmth, hospitality, encouragement, and genuine excitement.",
            "Welcome new hire with warm hospitality and team excitement.",
            [], ["warm", "hospitable", "encouraging", "excited"],
            ["employee first day onboarding"], []
        ),
        (
            "p079",
            "Explain why a requested software feature cannot be built. Use a collaborative, transparent, empathetic, and respectful rationale.",
            "Explain software feature rejection collaboratively and empathetically.",
            [], ["collaborative", "transparent", "empathetic", "respectful"],
            ["product feature rejection decision"], []
        ),
        (
            "p080",
            "Write a farewell note to colleagues upon retiring after 35 years. Express humble nostalgia, heartfelt warmth, and quiet peace.",
            "Write retirement farewell note with warmth and humble nostalgia.",
            [], ["humble nostalgia", "heartfelt warmth", "quiet peace"],
            ["career retirement after 35 years"], []
        ),
    ]

    for pid, prompt, intent, constraints, tone, context, steps in tone_data:
        prompts.append(PromptItem(
            id=pid,
            category="tone",
            prompt=prompt,
            ground_truth=GroundTruth(
                intent=intent,
                role="communicator",
                constraints=constraints,
                tone=tone,
                context=context,
                steps=steps
            )
        ))

    # -------------------------------------------------------------
    # 5. MULTI-STEP PROMPTS (20 prompts: p081 - p100)
    # -------------------------------------------------------------
    multi_step_data = [
        (
            "p081",
            "First identify the problem. Then explain its cause. Then provide two possible solutions. Finally recommend the safer option.",
            "Execute four-stage problem analysis, causal explanation, solution ideation, and risk-averse selection.",
            [], ["methodical", "analytical"], ["operational problem scenario"],
            [
                "identify the problem",
                "explain its cause",
                "provide two possible solutions",
                "recommend the safer option"
            ]
        ),
        (
            "p082",
            "Step 1: Extract all numerical data from the paragraph. Step 2: Calculate the mean and variance. Step 3: Interpret what the variance reveals.",
            "Extract numbers, compute descriptive statistics, and interpret distributional variance.",
            [], ["mathematical", "procedural"], ["numerical data in paragraph"],
            [
                "extract all numerical data",
                "calculate mean and variance",
                "interpret variance implications"
            ]
        ),
        (
            "p083",
            "First parse the raw JSON payload. Next filter out expired sessions. After that group remaining sessions by user ID. Finally output the aggregated count per user.",
            "Process JSON sessions through parsing, filtering, user grouping, and frequency aggregation.",
            [], ["algorithmic", "precise"], ["user session payload"],
            [
                "parse raw JSON payload",
                "filter out expired sessions",
                "group sessions by user ID",
                "output aggregated count per user"
            ]
        ),
        (
            "p084",
            "Stage 1: Read the patient symptoms. Stage 2: List three differential diagnoses. Stage 3: Order diagnostic tests for each. Stage 4: Highlight critical contraindications.",
            "Conduct structured clinical differential diagnosis, diagnostic ordering, and contraindication review.",
            [], ["clinical", "rigorous"], ["patient clinical presentation"],
            [
                "read patient symptoms",
                "list three differential diagnoses",
                "order diagnostic tests for each diagnosis",
                "highlight critical contraindications"
            ]
        ),
        (
            "p085",
            "Begin by summarizing the customer complaint. Second, check our refund policy guidelines. Third, draft the response letter. Lastly, log the ticket resolution code.",
            "Execute complete customer support escalation and resolution workflow.",
            [], ["process-oriented", "professional"], ["customer complaint ticket"],
            [
                "summarize customer complaint",
                "check refund policy guidelines",
                "draft response letter",
                "log ticket resolution code"
            ]
        ),
        (
            "p086",
            "First identify the grammatical errors in this sentence. Second, correct them. Third, explain the grammatical rule that was violated.",
            "Identify syntax errors, apply grammatical corrections, and teach underlying syntax rules.",
            [], ["instructive", "pedagogical"], ["flawed sentence text"],
            [
                "identify grammatical errors",
                "correct grammatical errors",
                "explain violated grammatical rule"
            ]
        ),
        (
            "p087",
            "Step A: Isolate the suspect network subnet. Step B: Scan active IP addresses. Step C: Check open ports. Step D: Generate an alert report.",
            "Perform network security incident response quarantine, scan, and reporting workflow.",
            [], ["security-focused", "sequential"], ["compromised network environment"],
            [
                "isolate suspect network subnet",
                "scan active IP addresses",
                "check open ports on active IPs",
                "generate alert report"
            ]
        ),
        (
            "p088",
            "First brainstorm five startup concepts in climate tech. Then evaluate market size for each. Next pick the most viable idea. Finally outline a 30-day MVP roadmap.",
            "Brainstorm climate concepts, evaluate market sizes, select top idea, and formulate MVP execution plan.",
            [], ["entrepreneurial", "structured"], ["climate tech market opportunities"],
            [
                "brainstorm five startup concepts in climate tech",
                "evaluate market size for each concept",
                "select most viable idea",
                "outline 30-day MVP roadmap"
            ]
        ),
        (
            "p089",
            "1. Download the transaction logs. 2. Verify SHA-256 checksums. 3. Decrypt the log payload. 4. Parse anomalies into CSV format.",
            "Run secure audit log pipeline from download through integrity verification, decryption, and anomaly extraction.",
            [], ["technical", "procedural"], ["encrypted audit log repository"],
            [
                "download transaction logs",
                "verify SHA-256 checksums",
                "decrypt log payload",
                "parse anomalies into CSV format"
            ]
        ),
        (
            "p090",
            "Phase 1: Identify all stakeholders. Phase 2: Map their primary interests and conflicts. Phase 3: Propose a compromise resolution strategy.",
            "Conduct stakeholder discovery, conflicting interest mapping, and consensus mediation planning.",
            [], ["strategic", "diplomatic"], ["organizational restructuring dispute"],
            [
                "identify all stakeholders",
                "map primary interests and conflicts",
                "propose compromise resolution strategy"
            ]
        ),
        (
            "p091",
            "Start by inspecting the engine spark plugs. Then measure cylinder compression pressure. After that inspect fuel injector spray. Conclude with a fault summary.",
            "Diagnose internal combustion misfire through physical inspection, compression measurement, fuel test, and summary.",
            [], ["diagnostic", "mechanical"], ["engine misfire condition"],
            [
                "inspect engine spark plugs",
                "measure cylinder compression pressure",
                "inspect fuel injector spray pattern",
                "conclude with fault summary"
            ]
        ),
        (
            "p092",
            "Step 1: Ingest CSV records. Step 2: Remove rows with null timestamps. Step 3: Normalize currency columns to USD. Step 4: Write to database.",
            "Execute ETL pipeline data cleaning, currency conversion, and database persistence.",
            [], ["engineering", "sequential"], ["unclean financial CSV dataset"],
            [
                "ingest CSV records",
                "remove rows with null timestamps",
                "normalize currency columns to USD",
                "write clean records to database"
            ]
        ),
        (
            "p093",
            "First outline the plot structure. Then define character motivations. Next write the dialogue for the climactic confrontation. Finally review pacing.",
            "Structure narrative plot, establish psychological motivations, write climax dialogue, and refine pacing.",
            [], ["creative", "methodical"], ["screenplay scene concept"],
            [
                "outline plot structure",
                "define character motivations",
                "write climax dialogue",
                "review narrative pacing"
            ]
        ),
        (
            "p094",
            "1. Formulate a null hypothesis. 2. Choose an appropriate statistical significance test. 3. Compute p-value. 4. State whether to reject or fail to reject.",
            "Execute formal hypothesis test pipeline from hypothesis definition to test selection, calculation, and statistical decision.",
            [], ["statistical", "rigorous"], ["experimental treatment sample data"],
            [
                "formulate null hypothesis",
                "choose appropriate statistical test",
                "compute p-value",
                "state reject or fail to reject decision"
            ]
        ),
        (
            "p095",
            "Initial step: Clone the GitHub repository. Second step: Install dependencies with poetry. Third step: Execute pytest suite. Final step: Build Docker container.",
            "Execute CI/CD build lifecycle steps from repository cloning to testing and containerization.",
            [], ["DevOps", "ordered"], ["source code repository"],
            [
                "clone GitHub repository",
                "install dependencies with poetry",
                "execute pytest suite",
                "build Docker container"
            ]
        ),
        (
            "p096",
            "First assess immediate airway and breathing. Next stabilize cervical spine. Then check peripheral circulation. Finally prepare for patient transfer.",
            "Administer systematic trauma assessment prioritizing airway, spine stabilization, circulation, and transfer prep.",
            [], ["emergency medicine", "critical"], ["acute trauma victim"],
            [
                "assess airway and breathing",
                "stabilize cervical spine",
                "check peripheral circulation",
                "prepare for emergency patient transfer"
            ]
        ),
        (
            "p097",
            "Step 1: Identify all chemical reactants. Step 2: Balance the chemical equation. Step 3: Determine the limiting reagent. Step 4: Calculate theoretical yield in grams.",
            "Solve stoichiometric chemistry problem through reactant identification, balancing, limiting reagent finding, and yield calculation.",
            [], ["scientific", "computational"], ["chemical reaction quantities"],
            [
                "identify chemical reactants",
                "balance chemical equation",
                "determine limiting reagent",
                "calculate theoretical yield in grams"
            ]
        ),
        (
            "p098",
            "First audit website SEO metadata. Second analyze page load performance on mobile. Third inspect keyword ranking competition. Fourth deliver prioritized SEO recommendations.",
            "Conduct SEO website audit spanning metadata, mobile performance, competitor rankings, and prioritized recommendations.",
            [], ["marketing", "analytical"], ["client e-commerce website"],
            [
                "audit website SEO metadata",
                "analyze mobile page load performance",
                "inspect competitor keyword rankings",
                "deliver prioritized SEO recommendations"
            ]
        ),
        (
            "p099",
            "Part 1: Transcribe audio interview. Part 2: Strip identifying personally identifiable information. Part 3: Code recurring thematic patterns. Part 4: Synthesize qualitative findings.",
            "Conduct qualitative research workflow from transcription to anonymization, thematic coding, and findings synthesis.",
            [], ["research", "methodical"], ["recorded qualitative interviews"],
            [
                "transcribe audio interview",
                "strip personally identifiable information",
                "code recurring thematic patterns",
                "synthesize qualitative findings"
            ]
        ),
        (
            "p100",
            "Step 1: Verify user identity via two-factor authentication. Step 2: Check current account authorization permissions. Step 3: Execute password reset. Step 4: Dispatch confirmation notification.",
            "Execute secure user credential reset verification and dispatch workflow.",
            [], ["security", "procedural"], ["user account recovery request"],
            [
                "verify identity via two-factor authentication",
                "check account authorization permissions",
                "execute password reset",
                "dispatch confirmation notification"
            ]
        ),
    ]

    for pid, prompt, intent, constraints, tone, context, steps in multi_step_data:
        prompts.append(PromptItem(
            id=pid,
            category="multi_step",
            prompt=prompt,
            ground_truth=GroundTruth(
                intent=intent,
                role="analyst",
                constraints=constraints,
                tone=tone,
                context=context,
                steps=steps
            )
        ))

    return PromptDataset(
        version="1.0.0",
        description="Semantic Collapse Benchmark - 100 Balanced Curated Prompts across 5 Semantic Dimensions",
        prompts=prompts
    )


def main():
    print("Generating 100 benchmark prompts...")
    dataset = generate_prompts()
    
    # Validate
    validation = validate_dataset(dataset, expected_total=100)
    print("Validation result:", validation)
    if not validation["valid"]:
        raise ValueError(f"Dataset validation failed: {validation['errors']}")

    # Save to data/raw/prompts_100.json
    raw_path = Path("data/raw/prompts_100.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"Saved raw prompts to {raw_path}")

    # Save to data/processed/prompts.json
    proc_path = Path("data/processed/prompts.json")
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(proc_path, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"Saved processed prompts to {proc_path}")

    # Save ground truth annotations to data/annotations/ground_truth.json
    annotations = {
        item.id: {
            "category": item.category,
            "prompt": item.prompt,
            "ground_truth": item.ground_truth.model_dump()
        }
        for item in dataset.prompts
    }
    annot_path = Path("data/annotations/ground_truth.json")
    annot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(annot_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved annotations to {annot_path}")
    print("All 100 prompts successfully created and verified!")


if __name__ == "__main__":
    main()
