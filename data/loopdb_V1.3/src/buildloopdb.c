/************************************************************************/
/**

   \file       buildloopdb.c
   
   \version    V1.3
   \date       12.12.17
   \brief      Build a database of CDR-H3 like loops
   
   \copyright  (c) Dr. Andrew C. R. Martin, UCL, 2015-2017
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
   Reads a directory of PDB files and identifies stretches that match
   the takeoff region distances for CDR-H3 loops (i.e. H92-H94 with
   H103-H105). The mean and standard deviation distances are stored in
   distances.h which is built automatically from a directory of PDB
   files. A table containing distance ranges may be used to override
   these defaults. Output is a file containing the PDB code, residue range
   loop length (residues between the takeoff regions) and the 9 distances.

**************************************************************************

   Usage:
   ======

**************************************************************************

   Revision History:
   =================
   V1.0   16.07.15  Original   By: ACRM
   V1.1   03.11.15  Now reads the file list then processes files
                    rather than doing both in one step
                    Added -v
   V1.2   10.12.15  Added BackboneComplete() to check that all backbone
                    atoms are present
   V1.3   12.12.17  Bug fix in BackboneComplete() and changed default
                    minimum length to 1 residue. Also fixed problem
                    with multi-chain PDBs where chains after the first
                    would be analyzed multiple times

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
#define MAXBUFF                160
#define MAX_CA_CA_DISTANCE_SQ   16.0  /* max CA-CA distance of 4.0A     */
#define MAX_BOND_DISTANCE_SQ     4.0  /* max bond length of 2.0A        */

/************************************************************************/
/* Globals
*/

/************************************************************************/
/* Prototypes
*/
int  main(int argc, char **argv);
BOOL ParseCmdLine(int argc, char **argv, char *infile, char *outfile,
                  int *minLength, int *maxLength, BOOL *isDirectory,
                  char *distTable, BOOL *verbose, int *limit);
void Usage(void);
int  RunAnalysis(FILE *out, PDB *pdb, int minLength, int maxLength, 
                 char *pdbCode, REAL minTable[3][3], REAL maxTable[3][3]);
void PrintResults(FILE *out, char *pdbCode, int separation, PDB *p[3], 
                  PDB *q[3], REAL distMat[3][3]);
void ProcessFile(FILE *in, FILE *out, int minLength, int maxLength,
                 char *pdbCode, REAL minTable[3][3], REAL maxTable[3][3],
                 BOOL verbose);
void ProcessAllFiles(FILE *out, char *dirName, int minLength, 
                     int maxLength, REAL minTable[3][3], 
                     REAL maxTable[3][3], BOOL verbose, int limit);
void PrintHeader(FILE *out, char *dirName);
void ReadDistanceTable(char *distTable, REAL minTable[3][3],
                       REAL maxTable[3][3]);
void SetUpMinMaxTables(REAL minTable[3][3], REAL maxTable[3][3]);
BOOL ChainIsIntact(PDB *start, PDB *end);
BOOL BackboneComplete(PDB *pdb);


/************************************************************************/
/*>int main(int argc, char **argv)
   -------------------------------
*//**
-  14.07.15 Original   By: ACRM
-  12.12.17 Changed default minimum length to 1 residue
*/
int main(int argc, char **argv)
{
   char infile[MAXBUFF],
        outfile[MAXBUFF],
        distTable[MAXBUFF];
   FILE *in         = stdin,
        *out        = stdout;
   int  minLength   = 1,
        maxLength   = 0,
        retval      = 0,
        limit       = 0;
   BOOL isDirectory = FALSE,
        verbose     = FALSE;
   REAL minTable[3][3],
        maxTable[3][3];

   /* Default distance ranges for CDR-H3                                */
   SetUpMinMaxTables(minTable, maxTable);

   if(!ParseCmdLine(argc, argv, infile, outfile, &minLength, &maxLength,
                    &isDirectory, distTable, &verbose, &limit))
   {
      Usage();
      return(0);
   }
   else
   {
      if(distTable[0])
      {
         ReadDistanceTable(distTable, minTable, maxTable);
      }

      if(isDirectory)
      {
         if(blOpenStdFiles(NULL, outfile, NULL, &out))
         {
            PrintHeader(out, infile);
            ProcessAllFiles(out, infile, minLength, maxLength, 
                            minTable, maxTable, verbose, limit);
            FCLOSE(out);
         }
      }
      else
      {
         if(blOpenStdFiles(infile, outfile, &in, &out))
         {
            char *pdbCode;
            pdbCode = blFNam2PDB(infile);
            ProcessFile(in, out, minLength, maxLength, pdbCode, 
                        minTable, maxTable, verbose);
            FCLOSE(in);
            FCLOSE(out);
         }
         else
         {
            return(1);
         }
      }
      
   }
         
   return(retval);
}

/************************************************************************/
/*>void PrintHeader(FILE *out, char *dirName)
   ------------------------------------------
   \param[in]   *out      Output file pointer
   \param[in]   *dirName  Name of directory being processed

   Prints a short header for the database file
*//**
-  14.07.15 Original   By: ACRM
*/
void PrintHeader(FILE *out, char *dirName)
{
   time_t tm;
   
   time(&tm);
   fprintf(out,"#PDBDIR: %s\n",dirName);
   fprintf(out,"#DATE:   %s\n",ctime(&tm));
}

   
/************************************************************************/
/*>void ProcessAllFiles(FILE *out, char *dirName, int minLength, 
                        int maxLength, REAL minTable[3][3], 
                        REAL maxTable[3][3], int limit)
   --------------------------------------------------------------
*//**
   \param[in]   *out       Output file pointer
   \param[in]   *dirName   Directory being processed
   \param[in]   minLength  Minimum loop length (0 = no limit)
   \param[in]   maxLength  Maximum loop length (0 = no limit)
   \param[in]   minTable   table of minimum distances
   \param[in]   maxTable   table of maximum distances
   \param[in]   limit      Max number of PDB files to process

   Steps through all files in the specified directory and processes them
   via calls to ProcessFile()

-  14.07.15 Original   By: ACRM
-  03.11.15 Now reads the file list first and then works through.
            It seemed to be failing, maybe because the directory was
            changing?
-  04.11.15 Added limit
*/
void ProcessAllFiles(FILE *out, char *dirName, int minLength, 
                     int maxLength, REAL minTable[3][3], 
                     REAL maxTable[3][3], BOOL verbose, int limit)
{
   DIR           *dp;
   struct dirent *dent;
   char          filename[MAXBUFF],
                 *fname;
   FILE          *in;
   STRINGLIST    *fileList = NULL,
                 *string;
   int           count     = 0;
   
   
   /* Get the list of PDB files                                         */
   if((dp=opendir(dirName))!=NULL)
   {
      while((dent=readdir(dp))!=NULL)
      {
         if(dent->d_name[0] != '.')
         {
            if(limit && (++count > limit))
               break;

            sprintf(filename,"%s/%s",dirName,dent->d_name);
            if((fileList = blStoreString(fileList, filename))==NULL)
            {
               fprintf(stderr,"Error (buildloopdb): No memory for file \
list.\n");
               exit(1);
            }
            
            if(verbose)
            {
               fprintf(stderr,"Listing: %s\n", filename);
            }
         }
      }
      closedir(dp);
   }
   
   /* now work through the file list processing each in turn            */
   for(string=fileList; string!=NULL; NEXT(string))
   {
      fname = string->string;
      
      if((in=fopen(fname, "r"))!=NULL)
      {
         char *pdbCode;
         pdbCode = blFNam2PDB(fname);
         fprintf(stderr,"Processing: %s\n", fname);
         ProcessFile(in, out, minLength, maxLength, pdbCode, 
                     minTable, maxTable, verbose);
         fclose(in);
      }
      else if(verbose)
      {
         fprintf(stderr,"Could not open file: %s\n", fname);
      }
      
   }

   blFreeStringList(fileList);
}

/************************************************************************/
/*>void ProcessFile(FILE *in, FILE *out, int minLength, int maxLength, 
                    char *pdbCode, REAL minTable[3][3], 
                    REAL maxTable[3][3], BOOL verbose)
   --------------------------------------------------------------------
*//**
   \param[in]   *in        Input file pointer (for PDB file)
   \param[in]   *out       Output file pointer
   \param[in]   minLength  Minimum loop length
   \param[in]   maxLength  Maximum loop length
   \param[in]   pdbCode    PDB code for this file
   \param[in]   minTable   table of minimum distances
   \param[in]   maxTable   table of maximum distances
   \param[in]   verbose    Verbose mode

   Obtains the PDB data and calls RunAnalysis() to do the real work

-  14.07.15 Original   By: ACRM
-  03.11.15 RunAnalysis() now returns number of loops found
-  10.12.15 Added check that PDB backbone has no missing atoms
*/
void ProcessFile(FILE *in, FILE *out, int minLength, int maxLength, 
                 char *pdbCode, REAL minTable[3][3], REAL maxTable[3][3],
                 BOOL verbose)
{
   PDB *pdb;
   int natoms;

   if((pdb = blReadPDBAtoms(in, &natoms))!=NULL)
   {
      if(BackboneComplete(pdb))
      {
         /* Extract the CAs                                             */
         if((pdb = blSelectCaPDB(pdb))!=NULL)
         {
            int nLoops;
            
            /* Run the analysis                                         */
            nLoops = RunAnalysis(out, pdb, minLength, maxLength, pdbCode, 
                                 minTable, maxTable);
            if(verbose)
               fprintf(stderr,"%d loops found\n", nLoops);
            
            FREELIST(pdb, PDB);
         }
         else if(verbose)
         {
            fprintf(stderr,"No CA atoms extracted\n");
         }
      }
      else
      {
         FREELIST(pdb, PDB);
      }
   }
   else if(verbose)
   {
      fprintf(stderr,"No atoms read from PDB file\n");
   }
}



/************************************************************************/
/*>BOOL ParseCmdLine(int argc, char **argv, char *infile, char *outfile,
                     int *minLength, int *maxLength, BOOL *isDirectory,
                     char *distTable, BOOL *verbose, int *limit)
   ---------------------------------------------------------------------
*//**
   \param[in]   argc              Argument count
   \param[in]   **argv            Argument array
   \param[out]  *infile           Input filename (or blank string)  
   \param[out]  *outfile          Output filename (or blank string) 
   \param[out]  *minLength        miniumum loop length to display   
   \param[out]  *maxLength        maxiumum loop length to display   
   \param[out]  *isDirectory      Input 'filename' is a directory   
   \param[out]  *distTable        Distance table filename           
   \param[out]  *verbose          Verbose mode                      
   \param[out]  *limit            Max number of PDBs to process (0=all)
   \return                        Success

   Parse the command line

-  14.07.15 Original    By: ACRM
-  04.11.15 Added -v and -l
-  12.12.17 Changed default minimum length to 1
*/
BOOL ParseCmdLine(int argc, char **argv, char *infile, char *outfile,
                  int *minLength, int *maxLength, BOOL *isDirectory,
                  char *distTable, BOOL *verbose, int *limit)
{
   BOOL gotArg = FALSE;
   
   argc--;
   argv++;
   
   infile[0]    = outfile[0] = '\0';
   *minLength   = 1;
   *maxLength   = 0;
   *isDirectory = TRUE;
   distTable[0] = '\0';
   *limit       = 0;
   
   while(argc)
   {
      if(argv[0][0] == '-')
      {
         switch(argv[0][1])
         {
         case 'm':
            argv++;
            argc--;
            if(!argc || !sscanf(argv[0], "%d", minLength))
               return(FALSE);
            break;
         case 'x':
            argv++;
            argc--;
            if(!argc || !sscanf(argv[0], "%d", maxLength))
               return(FALSE);
            break;
         case 'l':
            argv++;
            argc--;
            if(!argc || !sscanf(argv[0], "%d", limit))
               return(FALSE);
            break;
         case 't':
            argv++;
            argc--;
            if(!argc)
               return(FALSE);
            strncpy(distTable, argv[0], MAXBUFF);
            break;
         case 'p':
            *isDirectory = FALSE;
            break;
         case 'v':
            *verbose = TRUE;
            break;
         default:
            return(FALSE);
            break;
         }
      }
      else
      {
         /* Check that there are only 1 or 2 arguments left             */
         if(argc > 2)
            return(FALSE);

         gotArg = TRUE;
         
         /* Copy the first to infile                                    */
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

   /* If it's a directory then we MUST have a directory name            */
   if(*isDirectory && !gotArg)
      return(FALSE);
   
   return(TRUE);
}


/************************************************************************/
/*>void Usage(void)
   ----------------
*//**
   Prints a usage message 

-  14.07.15 Original   By: ACRM
-  10.12.15 V1.2
-  12.12.17 V1.3
*/
void Usage(void)
{
   fprintf(stderr,"\nbuildloopdb V1.3 (c) 2015-17 UCL, Dr. Andrew C.R. \
Martin.\n");

   fprintf(stderr,"\nUsage: buildloopdb [-v][-m minLength][-x maxLength]\
[-t disttable]\n");
   fprintf(stderr,"                   [-l limit] pdbdir [out.db]\n");
   fprintf(stderr,"--or--\n");
   fprintf(stderr,"       buildloopdb -p [-v][-m minLength][-x maxLength]\
[-t disttable]\n");
   fprintf(stderr,"                   [in.pdb [out.db]]\n");
   

   fprintf(stderr,"\n                   -p Argument is a PDB file\n");
   fprintf(stderr,"                   -m Set minimum loop length [1]\n");
   fprintf(stderr,"                   -x Set maximum loop length \
[None]\n");
   fprintf(stderr,"                   -t Specify a distance table\n");
   fprintf(stderr,"                   -l Limit the number of PDB files\n");
   fprintf(stderr,"                   -v Verbose\n");

   fprintf(stderr,"\nReads a directory of PDB files and identifies \
stretches that match\n");
   fprintf(stderr,"the takeoff region distances for CDR-H3 loops (i.e. \
H92-H94 with\n");
   fprintf(stderr,"H103-H105). The mean and standard deviation distances \
are stored in\n");
   fprintf(stderr,"distances.h which is built automatically from a \
directory of PDB\n");
   fprintf(stderr,"files. Output is a file containing the PDB code, \
residue range\n");
   fprintf(stderr,"loop length (residues between the takeoff regions) \
and the 9 distances.\n");
   fprintf(stderr,"-t allows the default distance ranges to be \
overridden; the distance file\n");
   fprintf(stderr,"contains nine min/max distance pairs representing \
n0-c0, n0-c1, n0-c2,\n");
   fprintf(stderr,"n1-c0, n1-c1, n1-c2, n2-c0, n2-c1, n2-c2\n");

   fprintf(stderr,"\n-p is primarilly for testing - it builds a database \
from a single PDB\n\n");
   fprintf(stderr,"file instead of a directory of PDB files\n");

   fprintf(stderr,"\nInput/output is to standard input/output if files \
are not specified.\n");
   fprintf(stderr,"However without the -p flag, a directory name is not \
optional.\n\n");
}

/************************************************************************/
/*>int RunAnalysis(FILE *out, PDB *pdb, int minLength, int maxLength, 
                   char *pdbCode, REAL minTable[3][3], 
                   REAL maxTable[3][3])
   --------------------------------------------------------------------
*//**
   \param[in]   *out        Output file pointer
   \param[in]   *pdb        Pointer to PDB linked list
   \param[in]   minLength   Minimum loop length       
   \param[in]   maxLength   Maximum loop length       
   \param[in]   *pdbCode    PDB code for this file    
   \param[in]   minTable    table of minimum distances
   \param[in]   maxTable    table of maximum distances

   Does the real work of analyzing a structure. Steps through N-ter and
   C-ter triplets of residues to find those that match the requirements
   of the minTable and maxTable distance matrices as well as any specified
   loop length requirements.

-  14.07.15 Original   By: ACRM
-  03.11.15 Now returns number of loops found
-  04.11.15 Now calls blFindNextChain() rather than blFindNextChainPDB()
            so the the first chain isn't terminated.
-  13.12.17 Added check on chain change when finding Nter and Cter
            residues (fixed bug with 2nd and subsequent chains being
            done multiple times).
*/
int RunAnalysis(FILE *out, PDB *pdb, int minLength, int maxLength, 
                char *pdbCode, REAL minTable[3][3], REAL maxTable[3][3])
{
   PDB  *n[3], *c[3],
        *chain,
        *nextChain;
   REAL distMat[3][3];
   int  i, j, 
        nloops = 0,
        separation;
   
   for(chain=pdb; chain!=NULL; chain=nextChain)
   {
      nextChain = blFindNextChain(chain);
      
      /* Find an N-terminal residue                                     */
      for(n[0]=chain; 
          n[0]!=NULL && n[0]!=nextChain;
          NEXT(n[0]))
      {

         /* And find the next two                                       */
         n[1] = (n[0])?n[0]->next:NULL;
         n[2] = (n[1])?n[1]->next:NULL;

         /* If all three are valid                                      */
         if((n[2] != NULL) && (n[2]->next != NULL))
         {
            separation = 0;
            
            /* Find a C-terminal residue                                */
            for(c[0]=n[2]->next->next; 
                c[0]!=NULL && c[0]!=nextChain;
                NEXT(c[0]))
            {
               /* If the spacing between N and Cter is too long or not
                  long enough, break out
               */
               separation++;
               if(maxLength && (separation > maxLength))
                  break;

               if(separation >= minLength)
               {
                  /* And find the next two                              */
                  c[1] = (c[0])?c[0]->next:NULL;
                  c[2] = (c[1])?c[1]->next:NULL;

                  /* If all three are valid                             */
                  if((c[1] != NULL) && (c[2] != NULL))
                  {
                     BOOL badDistance = FALSE;
                  
                     if(ChainIsIntact((PDB*)(n[0]), c[2]->next))
                     {
                        /* Create the distance matrix                   */
                        for(i=0; i<3; i++)
                        {
                           for(j=0; j<3; j++)
                           {
                              distMat[i][j] = DIST(n[i], c[j]);
                              
                              if((distMat[i][j] < minTable[i][j]) ||
                                 (distMat[i][j] > maxTable[i][j]))
                              {
                                 badDistance = TRUE;
                                 i=4; /* Break out of outer loop        */
                                 break;
                              }
                           }
                        }

                        if(!badDistance)
                        {
                           nloops++;
                           PrintResults(out, pdbCode, separation, n, c, 
                                        distMat);
                        }
                        
                     }
                  }
               }
            }  
         }
      }
   }
   return(nloops);
}


/************************************************************************/
/*>BOOL ChainIsIntact(PDB *start, PDB *end)
   ----------------------------------------
*//**
   \param[in]    *start  Start of region
   \param[in]    *end    End of region
   \return               Is intact?

   Checks whether a chain is intact (i.e. doesn't have any chain breaks)

-  16.07.15 Original   By: ACRM
*/
BOOL ChainIsIntact(PDB *start, PDB *end)
{
   PDB *p;
   
   for(p=start; p!=end; NEXT(p))
   {
      if((p!=NULL) && (p->next != NULL))
      {
         if(DISTSQ(p, p->next) > MAX_CA_CA_DISTANCE_SQ)
         {
            return(FALSE);
         }
      }
   }
   
   return(TRUE);
}


/************************************************************************/
/*>void PrintResults(FILE *out, char *pdbCode, int separation, 
                     PDB *n[3], PDB *c[3], REAL distMat[3][3]) 
   ------------------------------------------------------------
*//**
   \param[in]   *out          Output file pointer
   \param[in]   *pdbCode      PDB code
   \param[in]   separation    loop length
   \param[in]   *n[]          N-ter three PDB pointers
   \param[in]   *c[]          C-ter three PDB pointers
   \param[in]   *distMat[][]  Distance matrix

   Prints the results for a loop already determined to match criteria

-  14.07.15 Original   By: ACRM
*/
void PrintResults(FILE *out, char *pdbCode, int separation, 
                  PDB *n[3], PDB *c[3], REAL distMat[3][3]) 
{
   char resid1[16],
        resid2[16];
   int  i, j;
               
   MAKERESID(resid1, n[0]);
   MAKERESID(resid2, c[2]);
   
   fprintf(out, "%s %s %s %d ", (pdbCode!=NULL)?pdbCode:"",
           resid1, resid2, separation);
   for(i=0; i<3; i++)
   {
      for(j=0; j<3; j++)
      {
         fprintf(out, "%.3f ", distMat[i][j]);
      }
   }
   fprintf(out, "\n");
}


/************************************************************************/
/*>void ReadDistanceTable(char *distTable, REAL minTable[3][3], 
                          REAL maxTable[3][3])
   ------------------------------------------------------------
*//**
   \param[in]   *distTable    Distance table filename
   \param[out]  minTable[][]  table of minimum distances
   \param[out]  maxTable[][]  table of maximum distances

   Reads a user-specified distance matrix table instead of using the
   defaults coded in distances.h

-  14.07.15 Original   By: ACRM
*/
void ReadDistanceTable(char *distTable, REAL minTable[3][3],
                       REAL maxTable[3][3])
{
   FILE *fp = NULL;
   char buffer[MAXBUFF];
   int  i = 0,
        j = 0;
   
   if((fp=fopen(distTable, "r"))!=NULL)
   {
      while(fgets(buffer, MAXBUFF, fp))
      {
         char *chp;
         TERMINATE(buffer);
         if((chp = strchr(buffer, '#'))!=NULL)
            *chp = '\0';
         KILLTRAILSPACES(buffer);
         if(strlen(buffer))
         {
            sscanf(buffer,"%lf %lf", 
                   &(minTable[i][j]), &(maxTable[i][j]));
            if((++j)==3)
            {
               j=0;
               i++;
            }
         }
      }
      
      fclose(fp);
   }
}


/************************************************************************/
/*>void SetUpMinMaxTables(REAL minTable[3][3], REAL maxTable[3][3])
   ----------------------------------------------------------------
*//**
   \param[out]  minTable[][]  table of minimum distances
   \param[out]  maxTable[][]  table of maximum distances

   Initializes the minimum and maximum distance matrices based on
   mean and standard deviation distances in distances.h

-  15.07.15 Original   By: ACRM
*/
void SetUpMinMaxTables(REAL minTable[3][3], REAL maxTable[3][3])
{
   int i, j;

   #include "distances.h"

   for(i=0; i<3; i++)
   {
      for(j=0; j<3; j++)
      {
         minTable[i][j] = means[i][j] - sdMult * sds[i][j];
         maxTable[i][j] = means[i][j] + sdMult * sds[i][j];
      }
   }
}


/************************************************************************/
/*>BOOL BackboneComplete(PDB *pdb)
   -------------------------------
*//**
   \param[in]  *pdb   PDB linked list
   \return            Is the backbone complete?

   Checks whether the backbone is complete - thus rejecting CA-only files
   like 3ixx

-  10.12.15 Original  By: ACRM
*/
BOOL BackboneComplete(PDB *pdb)
{
   PDB  *p, 
        *start,
        *nextRes,
        *n     = NULL, 
        *ca    = NULL,
        *c     = NULL,
        *o     = NULL,
        *cPrev = NULL;
   char chain[blMAXCHAINLABEL];

   chain[0] = '\0';
   
   /* Step through the PDB file one residue at a time                   */
   for(start=pdb; start!=NULL; start=nextRes)
   {
      nextRes = blFindNextResidue(start);
      
      /* If the chain has changed we reset everything                   */
      if(!CHAINMATCH(start->chain, chain))
      {
         n = ca = c = o = cPrev = NULL;
         strncpy(chain, start->chain, blMAXCHAINLABEL);
      }
      

      /* Step through the residue finding the important atoms           */
      cPrev = c;
      for(p=start; p!=nextRes; NEXT(p))
      {
         if(!strncmp(p->atnam, "N   ", 4))
            n=p;

         if(!strncmp(p->atnam, "CA  ", 4))
            ca=p;

         if(!strncmp(p->atnam, "C   ", 4))
            c=p;

         if(!strncmp(p->atnam, "O   ", 4))
            o=p;
      }

      /* Either all atoms must be found (protein) or none of the atoms
         found (nucleic acid)
      */
      if((n==NULL) && (ca==NULL) && (c==NULL) && (o==NULL))     /* None */
      {
         continue;
      }
      else if((n!=NULL) && (ca!=NULL) && (c!=NULL) && (o!=NULL))/* All  */
      {
         /* Check the distances                                         */
         if(cPrev!=NULL)
         {
            if(DISTSQ(cPrev,n) > MAX_BOND_DISTANCE_SQ)
               return(FALSE);
         }
         if(DISTSQ(n,ca) > MAX_BOND_DISTANCE_SQ)
            return(FALSE);
         if(DISTSQ(ca,c) > MAX_BOND_DISTANCE_SQ)
            return(FALSE);
         if(DISTSQ(c,o)  > MAX_BOND_DISTANCE_SQ)
            return(FALSE);
      }
      else
      {
         return(FALSE);
      }
   }
   return(TRUE);
}

