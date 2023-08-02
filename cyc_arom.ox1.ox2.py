# Python script to determine the value of the AR descriptor for the Ox1 and Ox2 cyclic dimers (difficult cases where there are two [sd] and [ds] options)
#
# Input file: cyc.ox1.ox2.dat (list with names of dimers in question)
# Output: cyc.ox1.ox2.out (provides value of AR descriptors)
#
# D. Bosch and L. Blancafort, Universitat de Girona, July 2023
#
import sys
inf=open('cyc.ox1.ox2.dat','r')
outf=open('cyc.ox1.ox2.out','w')
fr1=['23.100' ,'34.100' ,'45.1x0' ,'67.10x' ,'12.x10' ,'17.x10' ,'23.010' ,'34.010' ,'67.01x' ,'12.x01' ,'17.x01' ,'23.001' ,'34.001' ,'45.0x1' ,'23.111' ,'34.111']
fr2=[['d','d'],['d','d'],['d'],    ['d'],    ['z','z'],['z','z'],['z','z'],['d','z'],['z'],    ['d','z'],['d','z'],['z','d'],['z','z'],['z']    ,['d','z'],['d','d']]
fr3=[['A','N'],['N','A'],['N'],    ['N'],    ['N','A'],['N','A'],['A','N'],['A','A'],['N'],    ['N','A'],['A','A'],['A','N'],['N','A'],['N']    ,['N','N'],['N','N']]
fr4=[[],       [],       ['ds'],   ['sd'],   [],       [],       [],       [],       ['sd'],   [],       [],       [],       [],       ['ds']   ,[],       []       ]
fr01=['23.111','34.111']
fr02=[['d','z'],['d','d']]
fr03=[['N','N'],['N','N']]
fr04=[[],[]]
for i in range(len(fr4)):
    if len(fr4[i])==0:
        fr4[i]=['sd','ds']
for i in range(len(fr04)):
    if len(fr04[i])==0:
        fr04[i]=['sd','ds']
nl=0
for line in inf:
    result=[]
    ls=line.strip().replace('-','.').split('.')
    f1=ls[2]+'.'+ls[5]
    f2=ls[3]+'.'+ls[6]
    inv=False
    if ls[3][0] > ls[3][1]:
        inv=True
        ls0=ls[3][1]+ls[3][0]
        ls[3]=ls0
        f2=ls[3]+'.'+ls[6]
    for i in range(len(fr1)):
        if fr1[i]==f1:
            i1=i
        if fr1[i]==f2:
            i2=i
    for i01 in range(len(fr4[i1])):
        conn1=fr4[i1][i01]
        res1=fr2[i1][i01]
        arom1=fr3[i1][i01]
        for i02 in range(len(fr4[i2])):
            if inv:
                conn2=fr4[i2][i02][1]+fr4[i2][i02][0]
            else:
                conn2=fr4[i2][i02]
            res2=fr2[i2][i02]
            arom2=fr3[i2][i02]
# acceptable combination is conn1==2
            if conn1==conn2:
# now chose 'z' or 'd'
                if res1=='d' and res2=='d':
                    res0='d'
                else:
                    res0='z'
# chose aromaticity
                arom0=arom1+arom2
                result.append([conn1,res0,arom0])
    if len(result)==1:
        Aromx=arom0
        Resx=result[0][1]
    else:
        Aromx='NN'
        Resx='z'
        if result[0][1]=='d':
            Resx='d'
            if result[1][1]=='d':
               if 'AA' in result[0][2] or 'AA' in result[1][2]:
                   Aromx='AA'
               elif 'A' in result[0][2] or 'A' in result[1][2]:
                   Aromx='AN'
            else:
                Aromx=result[0][2]
        else:
            if result[1][1]=='d':
                Resx='d'
                Aromx=result[1][2]
            else:
                if 'AA' in result[0][2] or 'AA' in result[1][2]:
                   Aromx='AA'
                elif 'A' in result[0][2] or 'A' in result[1][2]:
                   Aromx='AN'
    print ('Final',line.strip(),Aromx,Resx)
    outf.write('%s %s %s\n' % (line.strip(),Aromx,Resx))
# check whether the value of LS determined by this script coincides with the s/d/z nomenclature of the ACIE 2021 paper (as it needs to be)
    if Resx!=ls[0]:
        print ('CONFLICT!',Resx,ls[0])
inf.close()
outf.close()
sys.exit()
