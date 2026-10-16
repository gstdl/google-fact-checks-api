# Literature map for the X verified-spreader SNA spike

Supports `notebooks/04_x_verified_spreader_sna_spike.py`. Each paper is tied to the research question (RQ) it helps answer, with the finding worth citing and the limit to state alongside it.

## Research questions

| RQ | Question |
| --- | --- |
| RQ1 | Do a small number of verified accounts account for a large share of the spread of false or low-credibility claims? |
| RQ2 | Is an interaction network (retweet, quote, reply, mention) a valid way to find those accounts when follower graphs are unavailable? |
| RQ3 | What does "verified" mean on X today, and how should the flag be used? |
| RQ4 | How far can a query-conditioned export (Brandwatch) be trusted for network analysis? |
| RQ5 | Does the Indonesian context change the picture (buzzers, coordination, platform mix)? |
| RQ6 | Does anchoring on fact-checked claims bias what we observe, and does stance (spread vs. debunk) need labelling? |

## Paper-to-RQ map

"Read" = abstract or relevant passages checked in the downloaded PDF. "Snippet" = claim taken from a search summary only, so confirm it before citing.

| File | Citation | Primary RQ | Also informs | Verification |
| --- | --- | --- | --- | --- |
| `pierri-2023-vaccine-misinformation-superspreaders.pdf` | Pierri et al., "One year of COVID-19 vaccine misinformation on Twitter" (arXiv preprint, Aug 2022; published in JMIR 2023) | RQ1 | RQ3 | Read |
| `yang-2021-covid-infodemic-twitter-vs-facebook.pdf` | Yang et al., "The COVID-19 Infodemic: Twitter versus Facebook" (arXiv, 2021) | RQ1 | RQ3, RQ5 | Read |
| `deverna-2022-identifying-superspreaders.pdf` | DeVerna et al., "Identification and characterization of misinformation superspreaders on social media" (arXiv preprint, not peer reviewed) | RQ1 | RQ2 | Read |
| `vosoughi-2018-true-and-false-news.pdf` | Vosoughi, Roy and Aral, "The spread of true and false news online", Science 359(6380) | RQ6 | RQ1 | Read |
| `shao-2018-hoaxy-misinformation-network.pdf` | Shao et al., "Anatomy of an online misinformation network", PLOS ONE 2018 (arXiv version) | RQ2 | RQ1, RQ6 | Read |
| `drolsbach-2023-community-fact-checked-diffusion.pdf` | Drolsbach and Pröllochs, "Diffusion of Community Fact-Checked Misinformation on Twitter" (arXiv v4, 2023) | RQ6 | RQ1 | Read |
| `memon-2020-covid-misinformation-communities.pdf` | Memon and Carley, "Characterizing COVID-19 Misinformation Communities Using a Novel Twitter Dataset" (arXiv, 2020) | RQ6 | RQ2 | Read |
| `pierri-2020-diffusion-network-topology.pdf` | Pierri, Piccardi and Ceri, "Topology comparison of Twitter diffusion networks reliably reveals disinformation news" (arXiv v4, 2020) | RQ2 | RQ6 | Read |
| `stam-2024-dishonest-homogeneous-networks.pdf` | Stam et al., "DISHONEST: Dissecting misInformation Spread using Homogeneous sOcial NEtworks and Semantic Topic classification" (arXiv, 2024) | RQ2 | RQ6 | Snippet (edge-type argument) |
| `weber-2020-reliability-of-osn-data-for-sna.pdf` | Weber et al., "A method to evaluate the reliability of social media data for social network analysis" (arXiv, 2020) | RQ4 | RQ2 | Read (abstract) |
| `weber-2021-streamed-data-variations-sna.pdf` | Weber et al., "Exploring the effect of streamed social media data variations on social network analysis", Social Network Analysis and Mining (arXiv, 2021) | RQ4 | RQ2 | Read (abstract) |
| `pante-2025-beyond-interaction-patterns.pdf` | Pantè et al., "Beyond Interaction Patterns: Assessing Claims of Coordinated Inter-State Information Operations on Twitter/X" (arXiv, 2025) | RQ2 | RQ5 | Read (abstract) |
| `logan-2023-twitter-interactions-multilayer-network.pdf` | Logan, LaCasse and Lunday, "Social network analysis of Twitter interactions: a directed multilayer network approach", Social Network Analysis and Mining 13:65 (2023) | RQ2 | RQ4 | Read (abstract) |
| `mcguirk-2021-brandwatch-for-classrooms-review.pdf` | McGuirk, "Performing social media analytics with Brandwatch for Classrooms: a platform review", Journal of Marketing Analytics 9 (2021) | RQ4 | none | Read (abstract and data-limit passages) |
| `xiao-2023-account-verification-paid-enrollment.pdf` | Xiao et al., "Account Verification on Social Media: User Perceptions and Paid Enrollment", USENIX Security 2023 | RQ3 | RQ1 | Snippet (survey figures) |
| `sianipar-2021-buzzer-human-machine-agency.pdf` | Sianipar et al., "Buzzer Works on Human-Machine Agency Role in Securing Indonesia's Voters Precinct", SCITEPRESS 2021 | RQ5 | RQ2 | Read (abstract) |
| `subekti-2025-indonesia-2024-election-disinformation.pdf` | Subekti et al., "Social media and disinformation for candidates: the evidence in the 2024 Indonesian presidential election", Frontiers in Political Science 2025 | RQ5 | RQ6 | Read (abstract) |

## What each paper gives the spike

### RQ1: verified accounts and concentration of spread

- **Pierri et al.** Roughly 800 verified "superspreaders" accounted for about 35% of misinformation reshares on an average day in 2021, and the top account for over 13% of retweets. Closest match to the notebook: verified accounts, reshares, low-credibility content. Limit: English-language COVID-19 vaccine content, labelled at the source level.
- **Yang et al.** Superspreaders on both Twitter and Facebook were often associated with low-credibility sources and tended to be verified by the platform, with evidence of coordinated sharing. Limit: this is pre-2023 verification, when a check meant the platform had vetted the account.
- **DeVerna et al.** Frames two questions that map onto this spike: whether superspreaders can be identified reliably from widely available data (RQ1 of that paper) and who they are (RQ2 of that paper). Among accounts classified as political, 91% (49 of 55) were conservative, so the finding is specific to US data. Use it for method design (rank accounts, then test whether the ranking predicts later behaviour), not for Indonesian claims.

### RQ2: interaction networks as a method

- **Shao et al. (Hoaxy).** Builds a retweet diffusion network from claims and fact-checking links, then runs k-core decomposition. Fact-checking nearly disappears toward the core while social bots proliferate. Template for a claim-anchored network with a core-periphery reading.
- **Pierri, Piccardi and Ceri.** Diffusion network topology alone separates mainstream from disinformation news domains. Supports building one network per claim and comparing structure.
- **Stam et al.** Argues for retweet-only edges because replies, quotes and mentions often carry disagreement. This is the main justification for the stance caveat in notebook 04, which mixes all four edge types. Confirm the argument in the full text before citing.
- **Logan et al.** Models Twitter user interactions as a weighted, directed network. PageRank performed best of four measures used to rank individual influence, and community detection results were mixed across the two algorithms tried (greedy modular and Leiden). Notebook 04 ranks accounts by interactions received and betweenness, so PageRank is the obvious metric to add. It also found that a topically focused network gave more robust results when tweets came from broad search queries, which bears on how the Brandwatch queries should be written. Topic modelling was done with LDA on English tweets, so that part may not transfer to Indonesian.
- **Pantè et al.** Interaction patterns alone gave no evidence of coordination once a control dataset was added. Supports the notebook's statement that edges are interactions, not coordination.

### RQ3: what "verified" means

- **Xiao et al.** Surveys how people understand verification and what paid enrolment changes. Search summaries report that about half of respondents wrongly believe a blue check still requires proof of identity. Supports splitting the verified flag by follower tier.

### RQ4: data reliability

- **McGuirk.** Reviews Brandwatch for Classrooms, the educational edition, and reports its licence limits: ten queries at a time, downloads of up to 100,000 mentions per day, and up to two years of historical data. These are classroom-licence limits and may not match the licence behind the API key being provisioned, so confirm the real limits with the account owner. It is a teaching-oriented platform review, not a methods paper, and it does not describe network exports.
- **Weber et al. (2020 and 2021).** Compare datasets collected with different tools on Twitter and show that collection differences change network analysis results. Supports the "query-conditioned sample" limit and a check that Brandwatch's sample is complete enough for the graph metrics used.

### RQ5: Indonesian context

- **Sianipar et al.** Social network analysis of 328,474 tweets from the 2019 election found buzzer activity played a significant role, and human agency was more impactful than machine agency in creating trending topics. Qualitative interpretation plus six expert interviews, so treat it as context.
- **Subekti et al.** Qualitative case study of 2024 election disinformation. It finds disinformation patterns more dominant on video- and text-friendly channels such as Facebook, YouTube and TikTok. A reminder that X may not be the main channel for these claims in Indonesia, so a finding on X is partial.

### RQ6: fact-check anchoring and stance

- **Vosoughi et al.** About 126,000 cascades labelled by six fact-checking organisations. False news travelled farther and faster than true news, and bots spread both at similar rates. Justifies starting from fact-checked claims.
- **Drolsbach and Pröllochs.** Community fact-checked misleading posts received about 36.6% fewer retweets than not-misleading posts, and the choice of fact-check source changes which posts are studied. Direct warning that a sample built from one fact-checker (here, the Google Fact Check index) shapes the result.
- **Memon and Carley.** Separates misinformed users from informed users who call out misinformation. Supports adding a stance label (spread, debunk, neutral) before calling any account a spreader.

## Gaps

- **No paper found on Indonesian verified-account spreaders of fact-checked claims.** Several searches returned only election-era buzzer and disinformation work. This gap is the spike's contribution and worth stating explicitly.
- **No methods paper on Brandwatch exports for network analysis.** McGuirk's platform review is the only Brandwatch source in this set. It covers features and licence limits for the classroom edition and says nothing about building interaction networks, so edge extraction from Brandwatch fields remains untested.
- **Not papers, so not downloaded:** the Lowy Institute and CIPG pieces on Indonesian buzzer networks, and an Indonesian-language thesis on buzzer detection with social network analysis at <https://repository.upi.edu/122920/>.

## Handling notes

- Figures for Pierri et al., Vosoughi et al. and Drolsbach and Pröllochs were checked against the PDFs. Treat the Xiao et al. and Stam et al. claims marked "Snippet" as unchecked.
- Several entries are arXiv preprints. Cite the published versions where they exist (Pierri et al. in JMIR 2023, Shao et al. in PLOS ONE 2018).
- The PDFs are third-party copyrighted. Decide whether `papers/` belongs in version control or in `.gitignore`.
