### Download rna seq data with RTCGA package

# Load package
library("RTCGA")

# Check HTML
# browseVignettes("RTCGA")

# Check available data times. According to docu, Version 20151101 of RTCA.seq contains RNAseq datasets released 2015-11-01.
checkTCGA("Dates")

# Check cohorts i.e cancer types
# This run needs the loading od package 'dplyr' first for function %>%.
(cohorts <- infoTCGA() %>%
  rownames() %>%
  sub("-counts", "", x=.))


# Downloading RNAseq files for all cohorts

# Create directory
dir.create("data_RNAseq_RTCGA")

releaseDate <- "2015-11-01"
sapply(cohorts, function(element){
  tryCatch({
    downloadTCGA(cancerTypes = element,
                 dataSet = 'rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level',
                 destDir = "data_RNAseq_RTCGA",
                 date = releaseDate)},
    error = function(cond){
      cat("Error: Maybe there weren't rnaseq data for ", element, " cancer .\n")
    }
    )
})


# Shortening paths and directories
list.files("data_RNAseq_RTCGA") %>%
  file.path("data_RNAseq_RTCGA", .) %>%
  file.rename(to = substr(.,start=1, stop=50))



# Remove NA files
list.files("data_RNAseq_RTCGA") %>%
  file.path("data_RNAseq_RTCGA", .) %>%
  sapply(function(x){
    if (x=="data_RNAseq_RTCGA/NA")
      file.remove(x)
  })


# Remove unneeded "MANIFEST.txt" file from each cohort folder
list.files("data_RNAseq_RTCGA") %>%
  file.path("data_RNAseq_RTCGA", .) %>%
  sapply(function(x){
    file.path(x, list.files(x)) %>%
      grep(pattern = 'MANIFEST.txt', x=., value=TRUE) %>%
      file.remove()
  })


# Assign paths to files downloaded

list.files("data_RNAseq_RTCGA") %>%
  file.path("data_RNAseq_RTCGA", .) %>%
  sapply(function(y){
    file.path(y, list.files(y)) %>%
      assign(value = .,
             x = paste0(list.files(y) %>%
                        gsub(x=.,
                             pattern="\\..*",
                             replacement="") %>%
                          gsub(x=., pattern="-", replacement = "_"),
                        ".rnaseq.path"),
             envir=.GlobalEnv)
  })

# Reading data with special function readTCGA to read and transpose data automatically
ls() %>%
  grep("rnaseq\\.path", x=., value=TRUE) %>%
  sapply(function(element){
    tryCatch({
      readTCGA(get(element, envir=.GlobalEnv),
               dataType = "rnaseq") %>%
        assign(value = .,
               x=sub("\\.path", "", x=element),
               envir=.GlobalEnv)
    }, error = function(cond){
      cat(element)
    })
    invisible(NULL)
  }
  )

