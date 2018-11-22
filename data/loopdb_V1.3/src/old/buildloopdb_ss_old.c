/* Includes
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "bioplib/pdb.h"
#include "bioplib/general.h"
#include "bioplib/macros.h"
#include "sstruc.h"

/************************************************************************/
/* Defines and macros
*/
#define MAXBUFF 160

#define TOUPPER(x)                                                       \
   do {                                                                  \
      if(islower((x))) {                                                 \
         (x) = toupper((x));                                             \
      } } while(0)
   
/************************************************************************/
/* Globals
*/
BOOL sNewChain = FALSE;

/************************************************************************/
/* Prototypes
*/
int  main(int argc, char **argv);
BOOL ParseCmdLine(int argc, char **argv, char *infile, char *outfile,
                  BOOL *verbose, BOOL *summary);
void Usage(void);
void WriteSummary(FILE *out, PDB *pdbStart, PDB *pdbStop);
void WriteResults(FILE *out, PDB *pdbStart, PDB *pdbStop);
void RunAnalysis(FILE *out, PDB *pdb);
BOOL FindLoop(PDB *pdb, PDB **loopStart, PDB **loopStop);
PDB *FindLoopStart(PDB *pdb);
PDB *FindLoopEnd(PDB *pdb);
void RelabelSecStr(PDB *pdb);
void ProcessLoop(FILE *out, PDB *loopStart, PDB *loopStop);


/************************************************************************/
/*>int main(int argc, char **argv)
   -------------------------------
   19.05.99 Original   By: ACRM
   27.05.99 Added error return if blCalcSS out of memory
*/
int main(int argc, char **argv)
{
   char infile[MAXBUFF],
        outfile[MAXBUFF];
   FILE *in     = stdin,
        *out    = stdout;
   PDB  *pdb    = NULL;
   int  natoms,
        retval  = 0;
   BOOL verbose = FALSE,
        summary = FALSE;
   
   
   if(!ParseCmdLine(argc, argv, infile, outfile, &verbose, &summary))
   {
      Usage();
      return(0);
   }
   else
   {
      if(blOpenStdFiles(infile, outfile, &in, &out))
      {
         if((pdb = blReadPDBAtoms(in, &natoms))!=NULL)
         {
            PDB *start, *stop;
            
            /* Calculate secondary structure for each chain             */
            for(start=pdb; start!=NULL; start=stop)
            {
               stop=blFindNextChain(start);

               if(blCalcSecStrucPDB(start, stop, verbose) != 0)
               {
                  fprintf(stderr,"Warning: secondary structure \
calculation failed on chain %s\n", start->chain);
               }
            }
            
            /* Extract the CAs                                          */
            pdb = blSelectCaPDB(pdb);

            /* Run the analysis                                         */
            RunAnalysis(out, pdb);

            FREELIST(pdb, PDB);
         }
      }
      else
      {
         return(1);
      }
   }
         
   return(retval);
}


/************************************************************************/
/*>BOOL ParseCmdLine(int argc, char **argv, char *infile, char *outfile,
                     BOOL *verbose, BOOL *summary)
   ---------------------------------------------------------------------
   Input:   int    argc              Argument count
            char   **argv            Argument array
   Output:  char   *infile           Input filename (or blank string)
            char   *outfile          Output filename (or blank string)
            BOOL   *verbose            Verbose?
            BOOL   *summary          Display summary rather than XMAS?
   Returns: BOOL                     Success

   Parse the command line

   19.05.99 Original    By: ACRM
   21.05.99 Added summary
*/
BOOL ParseCmdLine(int argc, char **argv, char *infile, char *outfile,
                  BOOL *verbose, BOOL *summary)
{
   argc--;
   argv++;
   
   infile[0] = outfile[0] = '\0';
   *verbose = *summary = FALSE;
   
   while(argc)
   {
      if(argv[0][0] == '-')
      {
         switch(argv[0][1])
         {
         case 'v':
            *verbose = TRUE;
            break;
         case 's':
            *summary = TRUE;
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
   
   return(TRUE);
}



/************************************************************************/
/*>void Usage(void)
   ----------------
   19.05.99 Original   By: ACRM
   21.05.99 Added flags
*/
void Usage(void)
{
   fprintf(stderr,"\nss V1.0 (c) 1999, Inpharmatica, Ltd.\n");

   fprintf(stderr,"\nUsage: ss [-v][-s] [in.xmas [out.xmas]]\n");
   fprintf(stderr,"          -v Verbose mode - report dropped \
3rd Hbonds, etc.\n");
   fprintf(stderr,"          -s Summary - write a simple summary file \
rather than XMAS\n");

   fprintf(stderr,"\nCalculates secondary structure assignments \
according to the method of\n");
   fprintf(stderr,"Kabsch and Sander. Reads and writes XMAS format \
files. Input/output is\n");
   fprintf(stderr,"to standard input/output if files are not \
specified.\n\n");
}


/************************************************************************/
/*>void WriteSummary(FILE *out, PDB *pdb)
   --------------------------------------
   Write a summary file with the residue names and secondary structure

   21.05.99 Original   By: ACRM
*/
void WriteSummary(FILE *out, PDB *pdbStart, PDB *pdbStop)
{
   PDB *p;
   
   for(p=pdbStart; p!=pdbStop; p=blFindNextResidue(p))
   {
      fprintf(out, "%s%d%s %s %c\n",
              p->chain,
              p->resnum,
              p->insert,
              p->resnam,
              p->ss);
   }
}


/************************************************************************/
void WriteResults(FILE *out, PDB *pdbStart, PDB *pdbStop)
{
   WriteSummary(out, pdbStart, pdbStop);
}

/************************************************************************/
void RunAnalysis(FILE *out, PDB *pdb)
{
   PDB *loopStart,
       *loopEnd,
       *chain,
       *nextChain;

   for(chain=pdb; chain!=NULL; chain=nextChain)
   {
      /* This finds the next chain and terminates the current one       */
      nextChain = blFindNextChainPDB(chain);
      
      for(;;)
      {
         if(!FindLoop(chain, &loopStart, &loopEnd))
            break;
         
         fprintf(out,"\n\nStart:\n");
         
         blWritePDBRecord(out, loopStart);
         blWritePDBRecord(out, loopEnd);
         
         ProcessLoop(out, loopStart, loopEnd);
         chain = loopEnd;
      }
   }
}

/************************************************************************/
/* Identifies a region bounded on each side by three strand characters.
   
 */
BOOL FindLoop(PDB *pdb, PDB **loopStart, PDB **loopEnd)
{
   RelabelSecStr(pdb);

   *loopStart = FindLoopStart(pdb);
   *loopEnd   = FindLoopEnd(*loopStart);

   if((*loopStart != NULL) && (*loopEnd != NULL))
      return(TRUE);
   return(FALSE);
}


/************************************************************************/
/* Identifies a set of three strand characters following a 'loop' and
   returns the first of them
*/
PDB *FindLoopEnd(PDB *pdb)
{
   int ECount  = 0;
   PDB *prev  = NULL,
       *prev2 = NULL,
       *p;

   /* Skip the 3 strand residues at the base of the loop                */
   for(ECount=0; ECount<3; ECount++)
   {
      if(pdb!=NULL)
         NEXT(pdb);
   }
   
   /* Skip along till the next set of three strand residues             */
   ECount = 0;
   for(p=pdb; p!=NULL; NEXT(p))
   {
      if(p->ss == 'E')
      {
         ECount++;
         if((ECount == 3) && (prev2 != NULL))
         {
            return(prev2);
         }
      }
      else
      {
         ECount = 0;
      }

      prev2 = prev;
      prev  = p;
   }

   return(NULL);
}

/************************************************************************/
/* Finds a series of three strand characters followed by a non-strand
   character. Returns the first of the three strand characters
*/
PDB *FindLoopStart(PDB *pdb)
{
   int ECount = 0;
   PDB *prev  = NULL,
       *prev2 = NULL,
       *prev3 = NULL,
       *p;
    
   for(p=pdb; p!=NULL; NEXT(p))
   {
      /* Increment count of E characters                                */
      if(p->ss == 'E')
         ECount++;

      /* If this is not a strand                                        */
      if(p->ss != 'E')
      {
         /* If we have 3 preceeding E characters                        */
         if((ECount >= 3) && (prev3 != NULL))
         {
            return(prev3);
         }
         ECount = 0;
      }

      prev3 = prev2;
      prev2 = prev;
      prev  = p;
   }

   return(NULL);
}

      
   

/************************************************************************/
void RelabelSecStr(PDB *pdb)
{
   PDB *p;
   
   for(p=pdb; p!=NULL; NEXT(p))
   {
      TOUPPER(p->ss);
   }
}


/************************************************************************/
void ProcessLoop(FILE *out, PDB *loopStart, PDB *loopStop)
{
   
}

