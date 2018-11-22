CC = gcc 
COPT = -O3 
LIBS = -lm 
EXE  = buildloopdb scanloopdb finddist

BOBJS  = buildloopdb.o 
BLIBS  = bioplib/ReadPDB.o \
         bioplib/fsscanf.o \
         bioplib/OpenStdFiles.o \
         bioplib/padterm.o \
         bioplib/chindex.o \
         bioplib/FindResidue.o \
         bioplib/BuildConect.o \
         bioplib/strcatalloc.o \
         bioplib/PDBHeaderInfo.o \
         bioplib/GetPDBChainLabels.o \
         bioplib/prime.o \
         bioplib/array2.o \
         bioplib/throne.o \
         bioplib/GetWord.o \
         bioplib/stringcat.o \
         bioplib/stringutil.o \
         bioplib/IndexPDB.o \
         bioplib/hash.o \
         bioplib/WritePDB.o \
         bioplib/FNam2PDB.o \
         bioplib/SelectCaPDB.o \
         bioplib/FindNextChainPDB.o \
         bioplib/FindNextResidue.o \
         bioplib/StoreString.o \
         bioplib/FreeStringList.o \
         bioplib/StructurePDB.o

SOBJS  = scanloopdb.o
SLIBS  = bioplib/OpenStdFiles.o \
         bioplib/SelectCaPDB.o \
         bioplib/ReadPDB.o \
         bioplib/fsscanf.o \
         bioplib/StoreString.o \
         bioplib/chindex.o \
         bioplib/FindNextResidue.o \
         bioplib/padterm.o \
         bioplib/WritePDB.o \
         bioplib/FreeStringList.o \
         bioplib/FindResidue.o \
         bioplib/BuildConect.o \
         bioplib/PDBHeaderInfo.o \
         bioplib/GetPDBChainLabels.o \
         bioplib/IndexPDB.o \
         bioplib/stringcat.o \
         bioplib/stringutil.o \
         bioplib/strcatalloc.o \
         bioplib/array2.o \
         bioplib/GetWord.o \
         bioplib/throne.o \
         bioplib/prime.o \
         bioplib/hash.o \
         bioplib/ParseRes.o \
         bioplib/FindResidueSpec.o

FOBJS  = finddist.o
FLIBS  = bioplib/ReadPDB.o \
         bioplib/fsscanf.o \
         bioplib/chindex.o \
         bioplib/WritePDB.o \
         bioplib/SelectCaPDB.o \
         bioplib/FindResidueSpec.o \
         bioplib/FindResidue.o \
         bioplib/hash.o \
         bioplib/prime.o \
         bioplib/stringutil.o \
         bioplib/PDBHeaderInfo.o \
         bioplib/BuildConect.o \
         bioplib/IndexPDB.o \
         bioplib/throne.o \
         bioplib/FindNextResidue.o \
         bioplib/GetPDBChainLabels.o \
         bioplib/padterm.o \
         bioplib/GetWord.o \
         bioplib/stringcat.o \
         bioplib/strcatalloc.o \
         bioplib/ParseRes.o \
         bioplib/array2.o \
         bioplib/StoreString.o \
         bioplib/FreeStringList.o


all : $(EXE)

buildloopdb : $(BOBJS) $(BLIBS)
	$(CC) $(COPT) -o $@ $(BOBJS) $(BLIBS) $(LIBS)

buildloopdb.o : buildloopdb.c distances.h
	$(CC) $(COPT) -c -o $@ $<

scanloopdb : $(SOBJS) $(SLIBS)
	$(CC) $(COPT) -o $@ $(SOBJS) $(SLIBS) $(LIBS)

finddist : $(FOBJS) $(FLIBS)
	$(CC) $(COPT) -o $@ $(FOBJS) $(FLIBS) $(LIBS)

.c.o : 
	$(CC) $(COPT) -c -o $@ $<

distances.h :
	wget http://www.bioinf.org.uk/abs/abdb/Data/NR_CombinedAb_Chothia.tar.bz2
	wget http://www.bioinf.org.uk/abs/abdb/Data/NR_CombinedHv_Chothia.tar.bz2
	tar xvf NR_CombinedAb_Chothia.tar.bz2
	tar xvf NR_CombinedHv_Chothia.tar.bz2
	mkdir -p abdb
	mv NR_CombinedAb_Chothia/* abdb
	mv NR_CombinedHv_Chothia/* abdb
	./makedistances.pl > distances.h

install : $(EXE)
	mkdir -p ../bin
	cp $(EXE) ../bin

clean :
	\rm -f $(BOBJS) $(SOBJS) $(FOBJS) $(BLIBS) $(SLIBS) $(FLIBS)
	\rm -rf NR_Combined??_Chothia*
	\rm -rf abdb

distclean : clean
	\rm -f $(EXE)

