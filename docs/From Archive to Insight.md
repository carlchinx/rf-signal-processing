***

# From Archive to Insight: An AI-Augmented Re-Analysis of Legacy RF Filter Data

*What nine years and an AI partner did to RF filter analysis*

**Bandpass filter under test · April 2017 measurement data · Re-analysed March 2026**

***

> *Decluttering an old drive, I found something unexpected: RF S-parameter analysis from nearly a decade ago, conducted for a client whose identity is immaterial to what follows. What is not immaterial are the two questions the files immediately put to me.*
>
> *The first was a matter of pride. Could I still do this — load the files, perform the standard characterisation, extract the critical figures — without a tutorial, without hesitation, without reaching for half-forgotten notes? I ran the analysis. The fundamentals held. Tick.*
>
> *The second was more interesting. Given everything that has changed since 2017, what else could those files tell me? Not what they told me then — what they could reveal now, with an AI assistant as a research partner rather than a blank script and a deadline.*
>
> *That is the question this article is about.*

***

*Know Thy Data!*

The four .s2p files sitting on my hard drive represented measurements characterising 4 bandpass filter units. Each one a two-port S-parameter sweep of a bandpass filter, measured in April 2017. The frequency sweep extended from 57 MHz to 10 GHz, capturing 1,530 frequency points per production unit. The output were standard Touchstone format, .s2p files. They had been sitting on a hard drive, doing nothing, for the better part of a decade.

Now, RF and baseband engineering are magical for most folks. However, for the purposes of this article, I will stick to the level of detail that a technically literate but non-RF-specialist reader can follow. The goal is to communicate the essence of what the data was, what it showed, and how the analysis evolved — not to teach RF engineering from scratch.

> *To set the context for those not familiar with the RF domain, a bandpass filter is a fundamental RF component that allows signals within a specific frequency range to pass through while attenuating signals outside that range. The S-parameters (scattering parameters) describe how RF energy is transmitted and reflected by the device under test, providing critical information about its performance. S11 (return loss) and S21 (insertion loss) are the key parameters for characterising a filter's performance. S12 and S22 are typically negligible for a passive, reciprocal device like a bandpass filter, and were not the focus of the analysis. A port in this context is a physical interface where the RF signal enters or exits the device. For a two-port device, Port 1 is typically the input and Port 2 is the output. The S-parameters are defined as follows: S11 is the ratio of the reflected signal at Port 1 to the incident signal at Port 1; S21 is the ratio of the transmitted signal at Port 2 to the incident signal at Port 1; S12 is the ratio of the transmitted signal at Port 1 to the incident signal at Port 2; and S22 is the ratio of the reflected signal at Port 2 to the incident signal at Port 2. For a passive, reciprocal device, S12 is equal to S21, and S22 is equal to S11. The data in these files would allow for a comprehensive analysis of the filter's performance across its operating frequency range.*

The filter's job is deceptively simple: pass signals in one narrow frequency window and obliterate everything else. In 2017, I did what every competent RF engineer does. I processed the files using custom C++ code, plotted insertion loss and return loss, extracted the −3 dB passband, noted the stopband rejection figures, and wrote up a clean summary. Tick. Done. Deliverable submitted, client happy, invoice paid.

What I did not do — could not easily do, in the time available, with the tools available — was ask the data the deeper questions it was quietly waiting to answer.

Revisiting those same four files in 2026, equipped with an AI assistant as a research partner, the experience was something qualitatively different. Not just faster. Not just prettier plots. A fundamentally different relationship between practitioner and data — one where the question *"what else is in here?"* has a practical, same-session answer. That is the substance of this article.

What follows is an honest account of what happened and what is possible given a set of .s2p files and a capable AI assistant. The short version: *the data had far more to say than I asked it to say the first time.*

***

## Part One: The Tick — Establishing the Baseline

The discovery of those Touchstone files was, in an era where AI-augmented analysis is becoming the standard, an opportunity to pressure-test the evolution of the RF engineering workflow. Before layering on any advanced modelling capabilities, there is a profound necessity for a sanity check: verifying that the practitioner–AI partnership is built on a substrate of professional fluency. Verified foundational skills are the essential prerequisite.

The first test I set myself was therefore straightforward. Could I still process the files, extract the standard metrics, and produce a coherent characterisation? This is the bedrock of RF engineering competency — the fluency that separates practitioners from students. I wanted to know if that fluency was still there. This is not a trivial question. It is a necessary skill to ground myself and to verify that I am not just throwing tools at the problem, but that I can still do the basics without them and check if the results make sense.

As they say, I still got it - the skill is still there. The standard analysis came back without difficulty.

> **Stage 1 — Standard Characterisation** *· [interactive viewer](sparam_viewer-sanitised.html)*
>
> S21 and S11 overlaid across all four units, sweeping 57 MHz to 10 GHz; passband zoom to 3–5 GHz; per-unit comparison cards showing peak insertion loss, centre frequency, −3 dB bandwidth, and stopband rejection at 1 GHz (*See sparam_viewer-sanitised.html which contains traces togglable by unit, values readable on hover in GitHub*).
>
> This is the conventional deliverable. Competent, complete, and correct. It answered every question the client asked in 2017.

The findings were exactly what an experienced RF engineer would expect from a well-made bandpass filter. All four units pass a band centred between 3.7 and 4.2 GHz, with peak insertion losses between −0.60 and −0.83 dB. Stopband rejection at 1 GHz exceeded 100 dB across the batch, and return loss in the passband was adequate throughout.

**Key 2017 findings at a glance:**

| Parameter | Range across units |
| :-- | :-- |
| Peak S21 (Insertion Loss) | −0.60 to −0.83 dB |
| −3 dB bandwidth | 655–743 MHz |
| Stopband rejection at 1 GHz | −106 to −115 dB |
| Passband return loss (S11) | Better than −10 dB (typical) |

This is correct, professional engineering analysis. The standard practice exists because it efficiently extracts the information that matters for most use cases. The skill to do it fluently — to know what to look for, how to interpret the numbers, when a figure is concerning versus expected — is real and hard-won.

However, the standard practice also leaves the data mostly unread. A Touchstone file contains magnitude and phase at every frequency point — typically 1,500 or more. From that, one can compute group delay, phase linearity, time-domain behaviour, power balance, impedance trajectories, resonator properties, and statistical modes of variation across a production batch. In 2017, extracting any one of those would have required a dedicated script, calibrated expertise, a few hours, and a deliberate decision that the extra effort was warranted.

More directly: this standard characterisation provides zero diagnostic visibility into manufacturing process drifts or physical failure modes. It satisfies a professional baseline without uncovering the deeper story hidden within the hardware.

And then came the question that changed the session:

> *"This is good but it is not different from bog-standard analysis. Think of a post-doc researcher extending this visualisation work."*

That challenge is worth sitting with. The standard analysis is not trivial — it is the skill. Knowing which parameters matter, recognising when a figure is within expectation and when it is not, translating a sweep of numbers into a defensible engineering judgement: that is what the engagement was for, and what the 2017 deliverable represented. The challenge was not to replicate it. It was to go further — to ask the data the questions that the deliverable specification never required, but that the data was always capable of answering. What else is in there? What does the data know that I did not ask it to tell me? What does it say about the physics of the device, the manufacturing process, the real-world performance, and the inter-unit variation that the standard plots do not show? That is the question that set the next stage of analysis in motion.

***

## Part Two: Research-Level Characterisation and the Temporal Dimension

The first response to that challenge produced six analyses that do not appear in standard RF characterisation reports but sit firmly within the domain of what an experienced practitioner knows about and can interpret. They are not exotic. They are not out of reach. They are simply the next layer of information that the data carries, waiting to be asked. Each one is a standard analysis in the RF research community, but not in the engineering community. The session time from the standard characterisation to the research-level characterisation was under an hour.

By incorporating temporal and energetic dimensions, this phase transforms our understanding of the device from a black-box attenuator into a complex physical system — a transition vital for modern wideband system design, where the impact on signal integrity depends as much on time-domain behaviour as on frequency-domain rejection.

> **Stage 2 — Research-Level Characterisation** *· [interactive viewer](research_viz_sanitised.html)*
>
> Transmission & group delay dual-axis (A); time-domain impulse response via IFFT (B); power partition: transmitted / reflected / dissipated (C); manufacturing spread from ensemble mean (D); S11 impedance locus on Smith chart (E); S21 phase constellation polar plot (F). (*See research_viz_sanitised.html in GitHub for interactive viewer with hover annotations and toggles.*)
>
> All six panels are derived entirely from the original measurement data. No new measurements were required.

### Group Delay — The Invisible Dimension

The standard S21 plot shows amplitude. It does not show *time*. Group delay — the negative derivative of the unwrapped phase, $\tau_g(f) = -d\phi/d\omega$ — answers a question the magnitude response never can: how long does each frequency component spend inside the device? This is a critical question for any real-world signal, which is a superposition of many frequencies. A filter that passes the right frequencies but delays them by different amounts will distort the signal, even if the amplitude response looks good.

For the units under test, mid-band delay runs around 2–3 nanoseconds. At the band edges, it peaks at 5–7 nanoseconds. That variation is not a defect — it is physics. A filter with steep cut-off slopes must store and release energy at the transition frequencies; the time-bandwidth uncertainty principle, expressed in hardware. It means any wideband signal passing through this filter does not emerge intact — the edges of its spectrum arrive detectably later than the centre.

### Impulse Response — What a Perfect Pulse Sees

Taking the inverse Fourier transform of the complex S21 response — magnitude and phase together — reconstructs the time-domain behaviour: what the filter's output looks like if its input is a perfect, infinitely short impulse.

The inverse FFT (IFFT) is a standard numerical method for this transformation. The result is a complex time-domain trace that shows how the filter responds to a delta-function input. The impulse response is the filter's fingerprint in the time domain, revealing its internal energy storage and release dynamics.

The impulse responses show a consistent structure across all four units: a propagation delay of approximately 3 nanoseconds, followed by a ringing envelope that decays to 10% of its peak at around 12–14 nanoseconds. The ringing is not noise. It is the filter "remembering". Every joule it stores in its resonant cavities to achieve the frequency selectivity it was designed for has to go somewhere when the input stops.

What this communicates to a non-specialist is immediate. This device is not instantaneous. Feed it something short, and it will ring for over ten times the pulse duration. Every decibel of selectivity at the band edges is paid for in stored energy — a trade-off that becomes immediately visible when viewing the time-domain envelope.

### Power Partition — Where Every Watt Goes

Passivity imposes a strict conservation law: at every frequency, $|S_{21}|^2 + |S_{11}|^2 + \text{dissipation} = 1$. Decomposing the measured S-parameters into these three terms produces a power budget that no single-parameter plot conveys. *Passivity is a fundamental physical constraint on any real device. The power partition analysis makes it explicit. It shows how the incident power is divided among transmission, reflection, and dissipation at each frequency.*

In the passband, the partition runs approximately 57% transmitted, 30% reflected, and 13% dissipated as heat. In the stopband, the picture flips: over 95% of incident power is reflected, with little absorbed. The Smith chart trajectories and the S21 phase constellation polar plot confirm this: rejection is achieved primarily through reflection rather than absorption — a reactive, not absorptive, filter topology.

### Manufacturing Deviation — The Production Fingerprint

With four units measured, subtracting the ensemble mean S21 from each unit's response removes the shared characteristics and exposes only what makes each unit distinct. The result reveals the band edges as the most variable region, where small dimensional tolerances in the resonator coupling structure produce transmission changes of up to ±10 dB. The core passband, from roughly 3.7 to 4.4 GHz, is where the units agree most consistently, within about ±0.3 dB.

This is immediately useful intelligence for a production engineer. Not "units vary by 500 MHz," which the standard overlay already shows, but "units vary at the band edges because of edge-coupling tolerances, and the flat core is robust to whatever manufacturing spread you have." The resonator coupling structures are identified as the primary site of manufacturing sensitivity.

### Smith Chart — Reading the Impedance Story

Plotting S11 as a complex trajectory on the Smith chart rather than as magnitude versus frequency tells the impedance story across the full sweep. In the stopband, the locus hugs the outer rim, indicating near-total reflection. In the passband, the trace spirals inward toward the chart centre, indicating good matching.

The tightness of the four units' spirals in the passband confirms good impedance consistency across the production batch.

***

## Part Three: Deep Characterisation — Solving the Inverse Problem

> *"More visualisations and depth. Deeper analysis."*

The second push produced nine panels of analysis that move from characterisation into investigation — from measuring what the filter does to understanding *why*, and from describing the batch to decomposing it. The objective in this phase was to solve the "inverse problem": using the measured S-parameters to extract the internal physical parameters of the filter. This transition — from descriptive engineering to diagnostic inference — is where the depth of analysis begins to directly mitigate latent risks in high-reliability systems.

> **Stage 3 — Deep Characterisation** *· [interactive viewer](deep_sanitised.html)*
>
> Insertion loss & group delay with resonator poles marked (A); resonator pole map with loaded-Q (B); phase linearity residuals (C); Gaussian pulse propagation simulation (D); power partition (E); principal component analysis of inter-unit variation (F); stopband rejection at named wireless bands (G); S11 Smith chart with passband highlight (H); electrical length dispersion (I). (*See deep_sanitised.html in GitHub for interactive viewer with hover annotations and toggles.*)
>
> Derived entirely from the original four .s2p files. Session time from raw data to all three stages was under two hours.

### Resonator Pole Extraction — Reading the Architecture

Group delay peaks are not just features of a curve. They are *resonator poles* — the resonant frequencies of the individual coupled cavities inside the filter. Each peak carries two quantities: the resonant frequency $f_0$ and the loaded quality factor $Q_L$, computable from the peak height by $Q_L = \pi \cdot f_0 \cdot \tau_\text{peak}$.

A pole describes a resonator's contribution to the filter's overall response. The number of poles, their frequencies, and their Q values reveal the internal architecture of the filter — how many cavities it has, how they are coupled, and how lossy they are.

For the units under test, this analysis reveals a consistent two-pole structure, with poles appearing at approximately 3.7 GHz and 4.3 GHz and with unit-to-unit variation of about ±200–300 MHz. Loaded Q values run from roughly 59 to 91.

The key observation is the vertical spread in the pole map: the 500 MHz unit-to-unit centre-frequency variation is not a statistical artefact. It is the combined detuning of two resonator pairs shifting together — a systematic effect, not a random one.

### Phase Linearity Residuals — Quantifying Waveform Distortion

A filter with a perfectly linear phase response delays all signal components equally, preserving the waveform. Real bandpass filters are never perfectly linear in phase.

Fitting a straight line to the unwrapped S21 phase within the passband and plotting the residuals extracts the phase distortion directly. For all four units, the peak-to-peak phase ripple within the passband is 78–92°, with RMS values of 11–13°. This quantification is essential for predicting waveform degradation in modern digital modulation schemes.

To translate this for a non-specialist: a perfect filter would preserve the shape of any signal within its band. This filter does not — and the residuals say precisely how much it does not, and where the deviation is worst.

### Gaussian Pulse Propagation — Uncovering Latent Failure

A 600 MHz-bandwidth Gaussian RF pulse centred at 4 GHz was constructed in the frequency domain and multiplied by the complex S21 of each unit. The inverse FFT of the result gives the predicted output pulse.

> **Pulse propagation results — all four units**
>
> Units 1–3: **2.95–3.15 ns** delay and **−2.6 to −3.9 dB** insertion loss on the pulse peak.
>
> Unit 4: **3.30 ns** delay and **−6.53 dB** insertion loss, approximately 3 dB worse than the best unit.

The −6.53 dB figure for Unit 4 represents a latent failure that standard quality assurance would have missed entirely. The CW insertion loss at 4.2 GHz looks acceptable at approximately −0.63 dB, but the filter's offset centre frequency clips the edges of a 600 MHz pulse asymmetrically. A continuous-wave test gives Unit 4 a pass; a wideband pulse test reveals a 3 dB penalty against the rest of the batch. For a pulsed system with a tight link budget, that distinction matters.

### Batch Variation — PCA versus Physical Detuning Analysis

Quantifying inter-unit variation invites a comparison of two mathematical approaches, each with different diagnostic utility.

**Raw PCA:** Applying singular value decomposition to the 4 × 1,530 matrix of S21 responses decomposes the inter-unit variation into orthogonal modes.

> **PCA result**
>
> **93.4%** of all unit-to-unit variance is captured in a single mode, PC1.
>
> PC1 is an antisymmetric frequency-shift mode: units shift the passband up or down as a whole.
>
> PC2, at about 5%, is a bandwidth-ripple mode.
>
> Units 1 and 4 are the PC1 extremes, while Units 2 and 3 cluster in the middle.

The engineering implication of the PCA result is clear directionally: if you are tightening the production specification, most of the problem sits in a single process parameter — whatever dimension or material property sets the resonant frequency of the cavities.

**However**, PCA is a summary tool, not a diagnostic one. Its lack of physical semantics limits its utility: PC1 conflates bandwidth variation and frequency shifts without separating them into meaningful physical quantities.

**Shift-Fit Registration** is the superior physical model. By explicitly modelling frequency displacement — fitting each unit's response to the ensemble mean after applying a variable frequency offset — we achieve a 58.8% reduction in mean RMSE, dropping from 13.1 dB to 5.4 dB. This residual reduction directly quantifies how much of the observed variation is simply a rigid frequency shift versus a genuine shape change. The resulting detuning values are expressed in the engineer's native units:

| Unit | Detuning |
| :-- | :-- |
| Unit 1 | +184 MHz |
| Unit 2 | +70 MHz |
| Unit 3 | −62 MHz |
| Unit 4 | −188 MHz |

This tells a process engineer precisely what to target: not a statistical variance reduction, but a 188 MHz frequency calibration spread in the cavity resonators.

### Stopband Rejection at Named Wireless Bands

Rather than a generic "stopband rejection > 100 dB" figure, interpolating the measured S21 at specific named wireless frequencies produces a coexistence map that is directly legible to a system engineer.

| Band | Unit 1 | Unit 2 | Unit 3 | Unit 4 |
| :-- | :-- | :-- | :-- | :-- |
| Cellular 700 MHz | −103.5 dB | −104.0 dB | −102.7 dB | −101.1 dB |
| GSM 900 MHz | −111.6 dB | −112.5 dB | −104.6 dB | −104.0 dB |
| DCS 1.8 GHz | −113.4 dB | −105.5 dB | −105.6 dB | −108.6 dB |
| Wi‑Fi 2.4 GHz | −101.0 dB | −112.1 dB | −111.7 dB | −110.0 dB |
| 5G NR 5.0 GHz | −65.5 dB | −64.6 dB | −54.2 dB | −47.2 dB |
| X-band 7–8 GHz | −52 to −73 dB | −54 to −72 dB | −52 to −76 dB | −49 to −71 dB |
| Ku-band 10 GHz | −40.3 dB | −42.5 dB | −41.4 dB | −40.3 dB |

The degradation above 5 GHz is not a measurement artefact. It is the filter's harmonic window — a structural feature of coupled-resonator topologies in which partial transmission re-emerges at higher frequencies.

### Electrical Length Dispersion — The Speed of Phase

Electrical length — the normalised phase transit time, $-\angle S_{21} / (2\pi f)$ — measures how quickly the filter's phase structure propagates energy at each frequency. A non-dispersive medium would show a flat horizontal line.

The units under test show a pronounced downward slope across the passband: higher frequencies traverse the structure faster than lower ones. This is anomalous dispersion, and it is consistent with a coupled-resonator topology.

***

## The Numbers at a Glance

| Metric | Unit 1 | Unit 2 | Unit 3 | Unit 4 |
| :-- | :-- | :-- | :-- | :-- |
| Centre frequency | 3.713 GHz | 3.974 GHz | 4.072 GHz | 4.218 GHz |
| Peak S21 | −0.60 dB | −0.62 dB | −0.83 dB | −0.63 dB |
| −3 dB bandwidth | 743 MHz | 657 MHz | 684 MHz | 655 MHz |
| Mid-band group delay | ~2.6 ns | ~2.7 ns | ~3.0 ns | ~2.7 ns |
| Peak group delay (band edge) | ~5.3 ns | ~6.3 ns | ~6.2 ns | ~6.6 ns |
| Phase ripple (P-P) | 87.6° | 92.5° | 78.5° | 84.2° |
| Pulse IL @ 4 GHz | −2.95 dB | −2.62 dB | −3.92 dB | −6.53 dB |
| Resonator poles | 3.55, 4.28 GHz | 3.71, 4.33 GHz | 3.85, 4.52 GHz | 3.71, 4.00 GHz |
| Dominant $Q_L$ range | 59–91 | 74–80 | 74–84 | 81–83 |
| Passband dissipation | ~13% | ~13% | ~18% | ~13% |
| Frequency detuning (Shift-Fit) | +184 MHz | +70 MHz | −62 MHz | −188 MHz |
| PCA PC1 score | −524 | −213 | +202 | +535 |

***

## What Actually Changed — The Economics of Depth

The comparison is not simply "fewer outputs then, more outputs now." It is a change in the economics of depth. In 2017, the deep analysis presented here would have required a series of time-consuming project decisions and bespoke scripting. In 2026, it is a frictionless extension of engineering curiosity.

| | **2017 — Standard Practice** | **2026 — AI-Augmented Analysis** |
| :-- | :-- | :-- |
| Scope | Magnitude vs frequency; 2 parameters; standard deliverable | Magnitude, phase, group delay, time domain, power budget, impedance, statistical decomposition |
| Group delay | Not computed | Computed, smoothed, annotated with resonator pole markers |
| Phase | Not examined | 78–92° P-P residuals quantified per unit |
| Time domain | Not performed | Impulse response plus realistic pulse simulation |
| Power budget | Not performed | 57% transmitted / 30% reflected / 13% dissipated at mid-band |
| Batch variation | Visual range on overlay | PCA (93.4%); Shift-Fit Registration (detuning in MHz per unit) |
| Stopband context | "Exceeds 100 dB in stopband" | Named by wireless band; harmonic window at 5+ GHz quantified |
| Resonator physics | Implicit in bandwidth | Pole frequencies and loaded Q extracted analytically |
| Session time | Several hours, including scripting | Under two hours from raw files to all three stages |
| Audience | RF specialists only | Technical depth plus plain-language summary for any engineer |

The table shows what changed. What it cannot show is *why* it changed.

In 2017, every item on the right-hand column was theoretically possible. The physics was known. The mathematics was published. The numerical methods existed. The data was present. What was missing was not knowledge — it was the frictionless path from question to result. Writing a group-delay computation from scratch, debugging the phase unwrapping, computing the IFFT correctly, rendering it legibly, and then deciding to add the PCA and Shift-Fit analysis on top: each step was a discrete project decision, not a natural continuation of a train of thought.

> *The AI partner does not know more RF engineering than the practitioner. What it does is remove the toll booth between curiosity and result. The question ceases to be "is this worth the effort?" and becomes simply "what do I want to know?"*

***

## On Retained Skill

There is something worth dwelling on in that initial framing: *can I still do this the standard way?* The instinct to check — to verify competence against a known baseline — is healthy. The standard S-parameter analysis is the agreed vocabulary of the discipline.

If you cannot produce the standard outputs, you cannot participate in the standard conversations, and you cannot contextualise the deeper analysis or tell when the AI-assisted computation has produced something physically nonsensical. The fundamental skills are the substrate. The AI partnership is what you build on top of them.

Retained skill in a rapidly evolving tool landscape is more complex than it first appears. The practitioner who knew MATLAB in 2005 had skills. The practitioner who moved to Python in 2015 retained and extended them. The practitioner who learns to partner productively with AI in 2026 is not replacing skill — they are amplifying it. Domain knowledge remains the only safeguard against physically nonsensical results; the AI removes implementation friction, but it does not supply physical intuition.

What AI tools make obsolete is not domain expertise. It is the monopoly that implementation friction held over depth of analysis. In 2017, the decision not to compute group delay was often a reasonable resource-allocation choice. In 2026, it is simply a choice to leave information on the table.

***

## Implications for RF Engineering Practice

**The S-parameter file is the atomic unit of a much larger analysis.** The Touchstone format encodes complex information that the standard plots only partially surface. Group delay, time-domain behaviour, power budget, and impedance trajectories all live in those same files.

**Standard analysis can no longer be the destination.** While the standard characterisation remains the necessary starting point, it provides zero diagnostic visibility into manufacturing process drifts or physical failure modes. The latent failure in Unit 4 — invisible under standard CW testing, exposed only by the pulse simulation — is the clearest illustration of what remains in the data when the analysis stops at the conventional boundary.

**Communicating to non-specialists has always been the hard problem.** The 2017 deliverable was a report for experts. The 2026 analysis adds an interpretive layer that makes the physics legible to a system architect, a programme manager, or a procurement engineer who does not read Smith charts fluently.

**Production variation is a physics question, not just a statistics question.** The PCA result — that 93.4% of the inter-unit variance in this batch is a single frequency-shift mode — is a statement about the physical mechanism of manufacturing variability. The Shift-Fit Registration result makes it actionable: a 188 MHz calibration spread in cavity resonators, not a statistical abstraction.

**Historical data is more valuable than it appears.** The four .s2p files from 2017 were, from a 2017 perspective, a deliverable. From a 2026 perspective, they are a dataset — and one that was only partially analysed at the time. The analysis that would have taken a week in 2017 may take an afternoon in 2026.

***

## Conclusion: The Data Was Always Waiting

Four files. One afternoon. Three stages of analysis.

Stage 1 confirmed the tick — the standard competency is intact. Stage 2 added the temporal, energetic, and impedance story that the amplitude plots do not tell. Stage 3 went furthest: it solved the inverse problem, extracting the physical architecture of the device from its measured response, simulating real-signal performance, and decomposing eight years of untouched batch variation into a single actionable insight about the dominant manufacturing mode.

The device under test is, in the end, a well-designed filter. Good insertion loss, excellent lower stopband rejection, and consistent impedance matching. The 500 MHz centre-frequency spread across the batch is the headline finding from 2017, and it remains the headline finding now. What changed is everything around it: understanding that the spread is not a random statistical error but systematic resonator pair detuning (the PC1 mode); knowing that Unit 4's CW pass conceals a −6.53 dB pulse-peak penalty — a latent failure that would only manifest in a wideband pulsed system; and having a precise detuning map, in MHz, that tells a process engineer exactly what to calibrate.

The bog-standard analysis was the right starting point. It was never supposed to be the ending one. Every S-parameter file is the atomic unit of a much larger story. All data, even that gathered nearly a decade ago, carries untapped intelligence. The information was always there; we simply needed the right partner to help us ask the right questions.

***

*Data: four two-port S-parameter files (Touchstone .s2p format); measured 04 April 2017; 57 MHz–10 GHz; 1,530 points per unit; client and project details withheld; all derived quantities computed from original measurement data without additional measurements or calibration.*
