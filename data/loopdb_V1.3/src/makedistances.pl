#!/usr/bin/perl

for(my $i=0; $i<9; $i++)
{
    $Sx[$i] = $SxSq[$i] = $NValues[$i] = $mean[$i] = $sd[$i] = 0;
}

my $results = `for file in abdb/*; do ./finddist \$file; done`;

my @lines = split(/\n/, $results);
foreach my $line (@lines)
{
    my @fields = split(/\s+/, $line);

    for(my $i=0; $i<9; $i++)
    {
        CalcExtSD($fields[$i], 0, \$Sx[$i], \$SxSq[$i], \$NValues[$i], \$mean[$i], \$sd[$i]);
    }
    print STDERR "$line\n";
}

for(my $i=0; $i<9; $i++)
{
    CalcExtSD(0, 1, \$Sx[$i], \$SxSq[$i], \$NValues[$i], \$mean[$i], \$sd[$i]);
}

print "/*** This file auto-generated by makedistances.pl ***/\n\n";
print "REAL sdMult = 2.0;\n";
printf "REAL means[3][3] = {{%.6f, %.6f, %.6f},\n", $mean[0], $mean[1], $mean[2];
printf "                    {%.6f, %.6f, %.6f},\n", $mean[3], $mean[4], $mean[5];
printf "                    {%.6f, %.6f, %.6f}};\n", $mean[6], $mean[7], $mean[8];
printf "REAL sds[3][3]   = {{%.6f, %.6f, %.6f},\n", $sd[0], $sd[1], $sd[2];
printf "                    {%.6f, %.6f, %.6f},\n", $sd[3], $sd[4], $sd[5];
printf "                    {%.6f, %.6f, %.6f}};\n", $sd[6], $sd[7], $sd[8];



# Example usage:
# ==============
# use sd;
# my $Sx      = 0;
# my $SxSq    = 0;
# my $NValues = 0;
# my $mean    = 0;
# my $sd      = 0;
# my @values  = (1, 2, 3, 4, 5, 6);
# foreach my $value (@values)
# {
#    sd::CalcExtSD($value, 0, \$Sx, \$SxSq, \$NValues, \$mean, \$sd);
# }
# sd::CalcExtSD(0, 1, \$Sx, \$SxSq, \$NValues, \$mean, \$SD);
# printf "mean=%.3f SD=%.3f\n", $mean,  $SD;



#*************************************************************************
#   void CalcExtSD(REAL val, int action, REAL *Sx, REAL *SxSq, 
#                  int *NValues, REAL *mean, REAL *SD)
#   ----------------------------------------------------------
#   Calculate the mean and standard deviation from a set of numbers. 
#   The routine is called with each value to be sampled and the action 
#   required is specified:
#
#   Input:   val     int       The value to be sampled
#            action  short     0: Sample the value
#                              1: Calculate & return mean and SD
#                              2: Clear the sample lists
#   Output:  mean    *REAL     The returned mean
#            SD      *REAL     The returned standard deviation
#   I/O:     Sx      *REAL     Sum of values
#            SxSq    *REAL     Sum of values squared
#            NValues *int      Number of values
#
#   The output values are only set when action==1
#
#   This is the same as CalcSD except that the Sx, SxSq and NValues
#   variables are kept outside the function instead of being static
#   within the function
#
#   13.10.93 Original based on CalcSD   By: ACRM
#   22.06.94 Fixed for only one value supplied
#   22.11.99 Translated from C to Perl
#
sub CalcExtSD
{
    my($val, $action, $Sx, $SxSq, $NValues, $mean, $SD) = @_;

    if($action==0)
    {
        ($$NValues)++;
        $$SxSq += ($val * $val);
        $$Sx   += $val;
    }
    elsif($action==1)
    {
        $$mean = $$SD = 0.0;
        $$mean = ($$Sx) / ($$NValues)  if($$NValues > 0);
        $$SD   = sqrt(($$SxSq - (($$Sx) * ($$Sx)) / ($$NValues)) / ($$NValues - 1))  if($$NValues > 1);
    }
    else
    {
        $$SxSq    = 0.0;
        $$Sx      = 0.0;
        $$NValues = 0;
    }
}

1;


