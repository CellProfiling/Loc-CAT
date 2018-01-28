#This function adds a column to the IF_imgs file for pathways


import numpy as np 
import argparse
import csv


def main():
  
  argparser = argparse.ArgumentParser()
  argparser.add_argument('if_imgs_path')
  argparser.add_argument('reactome_path')
  args = argparser.parse_args()

  outname = args.if_imgs_path[0:-4]+'_pathways.csv'

  #if_imgs_path = '../data/IF_images_to_devin_v7.csv'
  reactome_lines = csv.reader(open(args.reactome_path),delimiter='\t')

  pathway_dict = dict()
  for rline in reactome_lines:
    #get the current pathway and gene
    ensg_id = rline[0]
    pathway = rline[3]
    #if we haven't hit that gene yet, create a pathway list 
    if not pathway_dict.has_key(ensg_id):
      pathway_dict[ensg_id] = []
    #add the pathway to the list for that gene
    pathway_dict[ensg_id].append(pathway)

  if_lines = csv.reader(open(args.if_imgs_path))
  header = if_lines.next()

  #find which column is the ensembl IDs for the IF_images file
  ensg_col = []
  for i,item in enumerate(header):
    #hack for now since I don't have numpy arrays 
    #ensg_col.append(item == 'ensembl_ids')
    if item == 'ensembl_ids':
      ensg_col = i
      break

  header_out = header
  header_out.append('pathways')
  outfile = open(outname,'w')
  print(header_out)
  for h in header_out:
    outfile.write(h)
    outfile.write(',')
  outfile.write('\b\n')
  #Go through IF_images file and add a column for pathways
  for line in if_lines:
    curr_ensg = line[ensg_col]
    out_line = line
    if pathway_dict.has_key(curr_ensg):
      out_line.append('"'+','.join(pathway_dict[curr_ensg])+'"')
    else:
      out_line.append('""')
    for l in out_line:
      outfile.write(l)
      outfile.write(',')
    outfile.write('\b\n')
    


if __name__ == '__main__':
  main()
