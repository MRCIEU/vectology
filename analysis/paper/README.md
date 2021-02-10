## Methods section for paper

### Comparison to other approaches for automated mapping to ontology

https://elswob.github.io/vectology-2/#comparison-to-other-approaches-for-automated-mapping-to-ontology

 
Use EBI GWAS data set
 - https://github.com/EBISPOT/EFO-UKB-mappings

Map to EFO using:
- OnToma (turn off zooma EBI file mapping)
- Zooma (turn off zooma EBI file mapping)
- BioSentVec (use GWAS preprocessing?)
- BERT-EFO
 
How to include top X results?
- Top 1,5,10
- Maybe calculate distance from predicted EFO to manual EFO using nxontology. And then compare distributions of these.
  - Create heatmap of EFO and nxtontology score for each model
- Compare cosine distance of top hit to nxontology distance
  - can this be used to help predict bad mappings
  - what is the distribution of nxontology scores for top 10 predictions? Can this be used to infer a consensus hit from top 10?
  - take top 5 nxontology distances, create weighted average and create density plot 

To Do:
- pandas profiling on final df to check for correlations 

 
### Comparison to other approaches for automated mapping to ontology
 
Percentage agreement between BERT-EFO and standard OpenGWAS trait-trait mappings.
- Look at sample of those that disagree
 
Create two matrices
- For each method:
	- Embed all OpenGWAS terms
	- Compare each to each

Compare the two matrices as above 

Models:
- BioSentVec
- BlueBERT
- BERT-EFO
- Google Universal Sentence Encoder

Benchmarks:
- BIOSSES
- MEDSTS
- https://github.com/ncbi-nlp/BLUE_Benchmark