''' Downloads chrM, chrY, chr21, chr22 alignments from UCSC Genome Browser. '''

import os
links = [
    "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz20way/maf/chrM.maf.gz",
    "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz20way/maf/chrY.maf.gz",
    "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz20way/maf/chr21.maf.gz",
    "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz20way/maf/chr22.maf.gz",
]
order = ["chrM.maf", "chrY.maf", "chr21.maf", "chr22.maf"]

for filename, link in zip(order, links):
    command = "wget -O - %s | gunzip -c > %s" % (link, filename)
    os.system(command)
