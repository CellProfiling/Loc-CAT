# This function adds a column to the IF_imgs file for pathways

import os.path
import argparse
import csv
import collections

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('if_img_path')
    argparser.add_argument('reactome_path')
    argparser.add_argument('--outname', type=str, default=None)
    args = argparser.parse_args()

    reactome_path = args.reactome_path
    if_img_path = args.if_img_path
    outname = args.outname
    if not outname:
        outname = os.path.splitext(if_img_path)[0] + '_pathways.csv'

    pathways = collections.defaultdict(list)

    for gene_line in open(reactome_path):
        gene_line = gene_line.split('\t')
        gene_line = [x.strip() for x in gene_line]
        id_ = gene_line[0]
        gene_pathway = gene_line[3]

        pathways[id_].append(gene_pathway)

    if_images = csv.DictReader(open(if_img_path))
    headers = if_images.fieldnames

    headers.append('pathways')
    out_writer = csv.DictWriter(open(outname, 'w'), fieldnames=headers)
    out_writer.writeheader()

    for if_line in if_images:
        id_ = if_line['ensembl_ids']
        if_line['pathways'] = ','.join(pathways[id_])
        out_writer.writerow(if_line)
