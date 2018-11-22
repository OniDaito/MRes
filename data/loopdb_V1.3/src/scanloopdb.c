/************************************************************************/
/**

   \file       scanloopdb.c
   
   \version    V1.1
   \date       17.07.15
   \brief      Scan a structure against the loop database
   
   \copyright  (c) Dr. Andrew C. R. Martin, UCL, 2015
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


**************************************************************************

   Usage:
   ======

**************************************************************************
   Revision History:
   =================
   V1.0   16.07.15  Original   By: ACRM
   V1.1   17.07.15  Added -l to allow loop length to be specified

*************************************************************************/
/* Includes
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <time.h>
#include <math.h>

#include "bioplib/pdb.h"
#include "bioplib/general.h"
#include "bioplib/macros.h"

/************************************************************************/
/* Defines and macros
*/
#define MAXBUFF   160
#define SMALLBUFF  16

#define DEF_TOLERANCE 1.0
#define DEF_STARTRES  "H95"
#define DEF_ENDRES    "H102"

typedef struct _loop
{
   struct _loop *next;
   char pdbcode[SMALLBUFF],
        startRes[SMALLBUFF],
        endRes[SMALLBUFF],
        buffer[MAXBUFF];
   REAL score;
}  LOOP;


/************************************************************************/
/* Globals
*/

/************************************************************************/
/* Prototypes
*/
int  main(int argc, char **argv);
BOOL ParseCmdLine(int argc, char **argv, char *infile, char *outfile,
                  char *dbFile, REAL *tolerance, char *startRes, 
                  char *endRes, int *numResult, int *loopLen);
void Usage(void);
LOOP *FindLoops(PDB *pdb, char *startRes, char *endRes, FILE *dbf, 
                REAL tolerance, int loopLen);
BOOL PrintLoops(FILE *out, LOOP *loops, int maxLoops);
LOOP *ScanMatrix(REAL distMat[3][3], int LoopLen, FILE *dbf, 
                 REAL tolerance);
static int cmpResults(const void *p1, const void *p2);
LOOP **IndexResults(LOOP *loops, int *nLoops);


/************************************************************************/
/*>int main(int argc, char **argv)
   -------------------------------
*//**
   Main program

-  14.07.15 Original   By: ACRM
-  17.07.15 Handles loop length
*/
int main(int argc, char **argv)
{
   char infile[MAXBUFF],
        outfile[MAXBUFF],
        dbFile[MAXBUFF],
        startRes[SMALLBUFF],
        endRes[SMALLBUFF];
   int  natoms,
        numResult = 0,
        loopLen = 0;
   REAL tolerance = DEF_TOLERANCE;
   LOOP *loops    = NULL;
   PDB  *pdb      = NULL;
   FILE *in       = stdin,
        *out      = stdout,
        *dbf      = NULL;


   if(!ParseCmdLine(argc, argv, infile, outfile, dbFile, &tolerance, 
                    startRes, endRes, &numResult, &loopLen))
   {
      Usage();
      return(0);
   }
   else
   {
      if(blOpenStdFiles(infile, outfile, &in, &out))
      {
         if((dbf = fopen(dbFile, "r"))!=NULL)
         {
            if((pdb = blReadPDBAtoms(in, &natoms))!=NULL)
            {
               if((pdb = blSelectCaPDB(pdb))!=NULL)
               {
                  if((loops = FindLoops(pdb, startRes, endRes, dbf, 
                                        tolerance, loopLen))!=NULL)
                  {
                     if(!PrintLoops(out, loops, numResult))
                     {
                        fprintf(stderr,"No memory to sort results\n");
                        return(1);
                     }

                  }
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
            fprintf(stderr,"Unable to open database file\n");
            return(1);
         }
      }
      else
      {
         fprintf(stderr,"Unable to open input/output files\n");
         return(1);
      }
   }
         
   return(0);
}


/************************************************************************/
/*>LOOP *FindLoops(PDB *pdb, char *startRes, char *endRes, FILE *dbf, 
                   REAL tolerance, int loopLen)
   ------------------------------------------------------------------
*//**
   \param[in]  *pdb       PDB linked list
   \param[in]  *startRes  Residue identifier for first residue
   \param[in]  *endRes    Residue identifier for last residue
   \param[in]  *dbf       File pointer for database file
   \param[in]  tolerance  Allowed tolerance for an individual distance
   \param[in]  loopLen    Desired loop length (0 = same as structure)
   \return                Linked list of loops that match the criteria

   Scans the relevant residues against the loop database

-  14.07.15 Original   By: ACRM
-  17.07.15 Handles loop length as a parameter
*/
LOOP *FindLoops(PDB *pdb, char *startRes, char *endRes, FILE *dbf, 
                REAL tolerance, int loopLen)
{
   PDB  *p,
        *pStartRes,
        *pEndRes,
        *n[3] = {NULL, NULL, NULL},
        *c[3] = {NULL, NULL, NULL};
   int  i, j;
   REAL distMat[3][3];
   LOOP *loops = NULL;


   if((pStartRes = blFindResidueSpec(pdb, startRes))==NULL)
      return(NULL);
   if((pEndRes = blFindResidueSpec(pdb, endRes))==NULL)
      return(NULL);
   
   /* If loop length not specified, see how long the one in the PDB file
      is and use that length
   */
   if(loopLen == 0)
   {
      for(p=pStartRes; p!=pEndRes->next; NEXT(p))
      {
         loopLen++;
      }
   }
   

   /* Find the 3 residues before the start of the loop                  */
   for(p=pdb; p!=pStartRes; NEXT(p))
   {
      n[0] = n[1];
      n[1] = n[2];
      n[2] = p;
   }

   /* Find the 3 residues after the end of the loop                     */
   c[0] = pEndRes->next;
   c[1] = (c[0] != NULL)?c[0]->next:NULL;
   c[2] = (c[1] != NULL)?c[1]->next:NULL;

   /* Build the distance matrix                                         */
   for(i=0; i<3; i++)
   {
      for(j=0; j<3; j++)
      {
         distMat[i][j] = DIST(n[i], c[j]);
      }
   }

   /* Scan the matrix against the database                              */
   loops = ScanMatrix(distMat, loopLen, dbf, tolerance);

   return(loops);
}


/************************************************************************/
/*>LOOP *ScanMatrix(REAL distMat[3][3], int loopLen, FILE *dbf, 
                    REAL tolerance)
   ------------------------------------------------------------
*//**
   \param[in]  distMat[][] distance matrix from our structure
   \param[in]  loopLen     Loop length
   \param[in]  *dbf        File pointer for database file
   \param[in]  tolerance   Allowed tolerance for an individual distance
   \return                 Linked list of loops that match the criteria

   Does the actual work of sanning the distance matrix for the residues
   in question against the database

-  14.07.15 Original   By: ACRM
*/
LOOP *ScanMatrix(REAL distMat[3][3], int loopLen, FILE *dbf, 
                 REAL tolerance)
{
   char buffer[MAXBUFF],
        pdbCode[SMALLBUFF],
        startRes[SMALLBUFF],
        endRes[SMALLBUFF];
   REAL thisMat[3][3];
   int  i, j,
        thisLoopLen;
   LOOP *loops = NULL,
        *l = NULL;


   while(fgets(buffer, MAXBUFF, dbf))
   {
      char *chp;
      TERMINATE(buffer);
      if((chp = strchr(buffer, '#'))!=NULL)
         *chp = '\0';
      KILLTRAILSPACES(buffer);
      if(strlen(buffer))
      {
         if(sscanf(buffer,"%s%s%s%d%lf%lf%lf%lf%lf%lf%lf%lf%lf",
                   pdbCode, startRes, endRes, &thisLoopLen,
                   &(thisMat[0][0]), &(thisMat[0][1]), &(thisMat[0][2]), 
                   &(thisMat[1][0]), &(thisMat[1][1]), &(thisMat[1][2]), 
                   &(thisMat[2][0]), &(thisMat[2][1]), &(thisMat[2][2])))
         {
            REAL score = 0.0;

            if(thisLoopLen == loopLen)
            {
               BOOL ok = TRUE;
               for(i=0; i<3; i++)
               {
                  for(j=0; j<3; j++)
                  {
                     REAL badness = ABS(distMat[i][j] - thisMat[i][j]);
                     if(badness > tolerance)
                     {
                        ok = FALSE;
                        i = j = 10;
                        break;
                     }
                     score += badness;
                  }
               }
               if(ok)
               {
                  if(loops == NULL)
                  {
                     INIT(loops, LOOP);
                     l = loops;
                  }
                  else
                  {
                     ALLOCNEXT(l, LOOP);
                  }

                  if(l==NULL)
                  {
                     FREELIST(loops, LOOP);
                     return(NULL);
                  }

                  /* Save the data                                      */
                  l->score = score;
                  strncpy(l->buffer,   buffer,   MAXBUFF);
                  strncpy(l->pdbcode,  pdbCode,  SMALLBUFF);
                  strncpy(l->startRes, startRes, SMALLBUFF);
                  strncpy(l->endRes,   endRes,   SMALLBUFF);
               }
            }
         }
      }
   }
   return(loops);
}


/************************************************************************/
/*>BOOL ParseCmdLine(int argc, char **argv, char *infile, char *outfile,
                     char *dbFile, REAL *tolerance, char *startRes, 
                     char *endRes, int *numResult, int *loopLen)
   ---------------------------------------------------------------------
*//**
   \param[in]  argc              Argument count
   \param[in]  **argv            Argument array
   \param[out] *infile           Input filename (or blank string) 
   \param[out] *outfile          Output filename (or blank string)
   \param[out] *dbFile           Database file to search
   \param[out] *tolerance        Max deviation on an individual distance
   \param[out] *startRes         Start residue ID
   \param[out] *endRes           End residue ID
   \param[out] *numResult        Number of results to print
   \param[out] *loopLen          Loop length (0 to use the same as in the
                                 input PDB file)
   \return                       Success

   Parse the command line

-  14.07.15 Original    By: ACRM
-  17.07.15 Added loopLen
*/
BOOL ParseCmdLine(int argc, char **argv, char *infile, char *outfile,
                  char *dbFile, REAL *tolerance, char *startRes, 
                  char *endRes, int *numResult, int *loopLen)
{
   BOOL gotArg = FALSE;
   
   argc--;
   argv++;
   
   strcpy(startRes, DEF_STARTRES);
   strcpy(endRes,   DEF_ENDRES);
   infile[0]  = outfile[0] = dbFile[0] = '\0';
   *tolerance = DEF_TOLERANCE;
   *numResult = *loopLen = 0;
   
   while(argc)
   {
      if(argv[0][0] == '-')
      {
         switch(argv[0][1])
         {
         case 't':
            argv++;
            argc--;
            if(!argc || !sscanf(argv[0], "%lf", tolerance))
               return(FALSE);
            break;
         case 'n':
            argv++;
            argc--;
            if(!argc || !sscanf(argv[0], "%d", numResult))
               return(FALSE);
            break;
         case 'l':
            argv++;
            argc--;
            if(!argc || !sscanf(argv[0], "%d", loopLen))
               return(FALSE);
            break;
         case 'r':
            argv++;
            argc--;
            if(!argc || !sscanf(argv[0], "%s", startRes))
               return(FALSE);
            argv++;
            argc--;
            if(!argc || !sscanf(argv[0], "%s", endRes))
               return(FALSE);
            break;
         default:
            return(FALSE);
            break;
         }
      }
      else
      {
         /* Check that there are 1-3 arguments left                     */
         if((argc < 1) || (argc > 3))
            return(FALSE);

         gotArg = TRUE;
         
         /* Copy the first to dbFile                                    */
         strcpy(dbFile, argv[0]);
         
         /* If there's another, copy it to infile                       */
         argc--;
         argv++;
         if(argc)
            strcpy(infile, argv[0]);
            
         /* If there's another, copy it to outfile                      */
         argc--;
         argv++;
         if(argc)
            strcpy(outfile, argv[0]);
            
         return(TRUE);
      }

      argc--;
      argv++;
   }

   if(!gotArg)
      return(FALSE);
   
   return(TRUE);
}


/************************************************************************/
/*>void Usage(void)
   ----------------
*//**
   Prints a usage message

-  14.07.15 Original   By: ACRM
-  17.07.15 Handles loop length
*/
void Usage(void)
{
   fprintf(stderr,"\nscanloopdb V1.1 (c) 2015 UCL, Dr. Andrew C.R. \
Martin.\n");

   fprintf(stderr,"\nUsage: scanloopdb [-l loopLen][-t tol][-n nresults]\
[-r startres endres]\n");
   fprintf(stderr,"                  loops.db [in.pdb [out.txt]]\n");
   fprintf(stderr,"\n                  loops.db - loop database from \
buildloopdb\n");
   fprintf(stderr,"                  -l - Specify the loop length \
[Default: same as the\n");
   fprintf(stderr,"                       input PDB file]\n");
   fprintf(stderr,"                  -t - Set tolerance for an \
individual distance [%.2f]\n", DEF_TOLERANCE);
   fprintf(stderr,"                  -n - Maximum number of results \
[unlimited]\n");
   fprintf(stderr,"                  -r - Set the boundaries of the \
loop [%s %s]\n", DEF_STARTRES, DEF_ENDRES);

   fprintf(stderr,"\nGenerates a set of loop candidates from the loop \
database. Scans the\n");
   fprintf(stderr,"specified loop (default CDR-H3) from the PDB file \
against the database\n");
   fprintf(stderr,"and generates a list of hits ranked by overall \
deviation of the 9\n");
   fprintf(stderr,"distances that are calculated around the adjoining 3 \
residues either\n");
   fprintf(stderr,"side of the loop. -t specifies the maximum deviation \
for any individual\n");
   fprintf(stderr,"distance.\n");
   fprintf(stderr,"Input/output is to standard input/output if files are \
not specified.\n\n");

}


/************************************************************************/
/*>BOOL PrintLoops(FILE *out, LOOP *loops, int maxLoops)
   -----------------------------------------------------
*//**
   \param[in]  *out     Output file pointer
   \param[in]  *loops   Linked list of loops
   \param[in]  maxloops Maxmimum number of loops to print
   \return              Success in allocating memory

   Prints the resulting loops sorted by their fit to the distance matrix

-  14.07.15 Original   By: ACRM
*/
BOOL PrintLoops(FILE *out, LOOP *loops, int maxLoops)
{
   LOOP **indx = NULL;
   int  nLoops = 0, 
        i;

   /* Create and sort an array to index the linked list                 */
   if((indx = IndexResults(loops, &nLoops))==NULL)
      return(FALSE);

   if(maxLoops == 0)
      maxLoops = nLoops;

   for(i=0; i<nLoops && i<maxLoops; i++)
   {
      fprintf(out, "%s : %f\n", indx[i]->buffer, indx[i]->score);
   }
   free(indx);
   return(TRUE);
}


/************************************************************************/
/*>static int cmpResults(const void *p1, const void *p2)
   -----------------------------------------------------
*//**
   \param[in]  *p1    Pointer to first LOOP pointer
   \param[in]  *p2    Pointer to second LOOP pointer
   \return            -1: First is smaller;
                       0: Values are equal;
                      +1: First is larger

   Comparison routine used by qsort()

-  14.07.15 Original   By: ACRM
*/
static int cmpResults(const void *p1, const void *p2)
{
   REAL s1, s2;

   /* The input values are pointers to the LOOP pointers. We therefore
      first cast them to the correct type (LOOP **), dereference this
      to get a LOOP pointer and then extract the score from this.
   */
   s1 = (*((LOOP **)p1))->score;
   s2 = (*((LOOP **)p2))->score;

   if(s1 < s2)
   {
      return(-1);
   }
   if(s2 < s1)
   {
      return(+1);
   }
   return(0);
}


/************************************************************************/
/*>LOOP **IndexResults(LOOP *loops, int *nLoops)
   ---------------------------------------------
*//**
   \param[in]  *loops  Linked list of LOOP structures
   \param[out] *nLoops Number of loops
   \return             Array of loop pointers (or NULL if malloc() fails)

   Performs an index sort on an array of pointers to the loop structures.
   Sorting is on the basis of the score stored in each loop structure.

-  14.07.15 Original   By: ACRM
*/
LOOP **IndexResults(LOOP *loops, int *nLoops)
{
   LOOP *l, 
        **indx;

   *nLoops = 0;

   /* Count the number of results                                       */
   for(l=loops; l!=NULL; NEXT(l))
      (*nLoops)++;

   if((indx = (LOOP **)malloc((*nLoops) * sizeof(LOOP *)))!=NULL)
   {
      *nLoops = 0;
      for(l=loops; l!=NULL; NEXT(l))
      {
         indx[(*nLoops)++] = l;
      }

      qsort(indx, *nLoops, sizeof(LOOP *), cmpResults);
   }

   return(indx);
}
