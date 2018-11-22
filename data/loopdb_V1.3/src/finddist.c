/************************************************************************/
/**

   \file       finddist.c
   
   \version    V1.0
   \date       15.07.15
   \brief      A quick routine to find the distance matrix for a given
               PDB file. Not designed to be used by the end user
   
   \copyright  (c) Dr. Andrew C. R. Martin, UCL, 1988-2015
   \author     Dr. Andrew C. R. Martin
   \par
               Institute of Structural & Molecular Biology,
               University College London,
               Gower Street,
               London.
               WC1E 6BT.
   \par
               andrew@bioinf.org.uk
               andrew.martin@ucl.ac.uk
               
**************************************************************************

   This code is NOT IN THE PUBLIC DOMAIN, but it may be copied
   according to the conditions laid out in the accompanying file
   COPYING.DOC.

   The code may be modified as required, but any modifications must be
   documented so that the person responsible can be identified.

   The code may not be sold commercially or included as part of a 
   commercial product except as described in the file COPYING.DOC.

**************************************************************************

   Description:
   ============
   This is a very quick routine to calculate the distance matrix for the
   takeoff region around CDR-H3. It is designed to be used only by
   makedistances.pl with input from a numbered antibody file.

**************************************************************************

   Usage:
   ======

**************************************************************************

   Revision History:
   =================
   V1.0   15.07.15  Original   By: ACRM

*************************************************************************/
/* Includes
*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "bioplib/pdb.h"

/************************************************************************/
/* Defines and macros
*/

/************************************************************************/
/* Globals
*/

/************************************************************************/
/* Prototypes
*/

/************************************************************************/
/*>int main(int argc, char **argv)
   -------------------------------
*//**
   Main program

   15.07.15   Original   By: ACRM
*/
int main(int argc, char **argv)
{
   FILE *fp;
   char *keyResN[] = {"H92",  "H93",  "H94"};
   char *keyResC[] = {"H103", "H104", "H105"};
   PDB *pdb, *pN[3], *pC[3];
   int i, j;

   if((argc < 2) || (!strncmp(argv[1], "-h", 2)))
   {
      fprintf(stderr,"\nUsage: finddist file.pdb\n");
      fprintf(stderr,"\nThis code is designed to be called by \
makedistances.pl\n\n");
      return(1);
   }
   
   if((fp = fopen(argv[1], "r"))!=NULL)
   {
      int natoms;
      
      if((pdb=blReadPDB(fp, &natoms))!=NULL)
      {
         if((pdb=blSelectCaPDB(pdb))!=NULL)
         {
            for(i=0; i<3; i++)
            {
               if((pN[i] = blFindResidueSpec(pdb, keyResN[i]))==NULL)
               {
                  fprintf(stderr,"Residue not found: %s (%s)\n",
                          keyResN[i], argv[1]);
                  return(1);
               }
               if((pC[i] = blFindResidueSpec(pdb, keyResC[i]))==NULL)
               {
                  fprintf(stderr,"Residue not found: %s (%s)\n",
                          keyResC[i], argv[1]);
                  return(1);
               }
            }

            for(i=0; i<3; i++)
            {
               for(j=0; j<3; j++)
               {
                  printf("%.3f ", DIST(pN[i], pC[j]));
               }
            }
            printf("\n");
         }
         else
         {
            fprintf(stderr,"No CA atoms extracted from PDB\n");
            return(1);
         }
         
      }
      else
      {
         fprintf(stderr,"No atoms read from PDB file\n");
         return(1);
      }
   }
   else
   {
      fprintf(stderr,"Unable to open PDB file\n");
      return(1);
   }
   
   return(0);
}
