# GTEx data

Link to GTEx portal for Bulk RNA-Seq data: https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression
1. Download the zip file: GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz
2. Download the clinical data file:  GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt
3. Download the tissue types data file:  GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt

4. Process and build the train/test GTEx datasets for all coding genes or both coding and landmark genes.
Arguments:
- PATH_GTEX: 'path_to_gtex_dataset.csv'
- PATH_TISSUES: 'path_to_tissues_dataset.csv'
- PATH_CLINICAL: 'path_to_clinical_dataset.csv'
- CODING: 'y' for coding genes only, otherwise 'n'
- LANDMARK: 'y' for landmark genes only, otherwise 'n'


> `python script_process_gtex.csv -path_gtex <PATH_GTEX> -path_tissues <PATH_TISSUES> -path_clinical <PATH_CLINICAL> -coding_genes <CODING> -only_landmark <LANDMARK>`

