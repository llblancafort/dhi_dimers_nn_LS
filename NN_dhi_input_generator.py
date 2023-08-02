# Python script to generate input descriptors for the different structures
# D. Bosch and L. Blancafort, Universitat de Girona, July 2023
#
# input: dhi_dimer_data.dat (file with the DHI dimer names, energy values, oscillator strength values)
# cyc.ox1.ox2.out (file with the AR descriptor value for the difficult cases, generated by cyc_arom.ox1.ox2.py script)
#
# output: files QBB.out	QBF.out	QBS.out
#
# generate input layer for QBB model
import sys
infile = open('dhi_dimer_data.dat','r')
conn = []
rox = [[0 for a in range(6)] for b in range(830)]
rbond = []
newdescriptor = []
oxdesc = []
aromaticity = []
nl = 0
pi = -1
cis = str('c')
trans = str('t')
new_aromaticity = []
listname = []
acd2 = []
acd3 = []
aromcycdim= open('cyc.ox1.ox2.out','r')
for line in aromcycdim:
    listname.append(line.rstrip().split()[0])
    acd2.append(line.rstrip().split()[1])
    acd3.append(line.rstrip().split()[2])
aromcycdim.seek(0)
for line in infile:
    nl += 1
    if nl==1:
        continue
    pi += 1
    name = line.rstrip().split()[0]
    new_arom = 0
    num = 0
    found = 0
    for el in listname:
        if name==el:
            found = 1
            if acd3[num]=='s':
                newdescriptor.append('0')
            elif acd3[num]=='d':
                newdescriptor.append('1')
            elif acd3[num]=='z':
                newdescriptor.append('2')
            if acd2[num]=='NN':
                new_aromaticity.append('0')
            elif acd2[num]=='AN' or acd2[num]=='NA':
                new_aromaticity.append('1')
            elif acd2[num]=='AA':
                new_aromaticity.append('2')
        num += 1
    if found==0:
        if 'lin' in name:
            if (name[0]=='s' and name[8]=='2' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='s' and name[8]=='3' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='s' and name[8]=='4' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='s' and name[8]=='7' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='d' and name[8]=='3' and name[13]=='1' and name[14]=='0' and name[15]=='0') or (name[0]=='d' and name[8]=='4' and name[13]=='0' and name[14]=='1' and name[15]=='0') or (name[0]=='d' and name[8]=='7' and name[13]=='0' and name[14]=='0' and name[15]=='1') or (name[0]=='z' and name[8]=='3' and name[13]=='0' and name[14]=='1' and name[15]=='0') or (name[0]=='z' and name[8]=='3' and name[13]=='0' and name[14]=='0' and name[15]=='1'):
                new_arom += 1
            if (name[0]=='s' and name[9]=='2' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='s' and name[9]=='3' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='s' and name[9]=='4' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='s' and name[9]=='7' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='d' and name[9]=='3' and name[17]=='1' and name[18]=='0' and name[19]=='0') or (name[0]=='d' and name[9]=='4' and name[17]=='0' and name[18]=='1' and name[19]=='0') or (name[0]=='d' and name[9]=='7' and name[17]=='0' and name[18]=='0' and name[19]=='1') or (name[0]=='z' and name[9]=='3' and name[17]=='0' and name[18]=='1' and name[19]=='0') or (name[0]=='z' and name[9]=='3' and name[17]=='0' and name[18]=='0' and name[19]=='1'):
                new_arom += 1
        elif 'cyc' in name:
            if (name[0]=='s' and ((name[6]=='1' and name[7]=='2') or (name[7]=='1' and name[6]=='2')) and name[14]=='x' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='1' and name[7]=='7') or (name[7]=='1' and name[6]=='7')) and name[14]=='x' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='2' and name[7]=='3') or (name[7]=='2' and name[6]=='3')) and name[14]=='0' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='3' and name[7]=='4') or (name[7]=='3' and name[6]=='4')) and name[14]=='0' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='4' and name[7]=='5') or (name[7]=='4' and name[6]=='5')) and name[14]=='0' and name[15]=='x' and name[16]=='0') or (name[0]=='s' and ((name[6]=='6' and name[7]=='7') or (name[7]=='6' and name[6]=='7')) and name[14]=='0' and name[15]=='0' and name[16]=='x') or (name[0]=='d' and ((name[6]=='3' and name[7]=='4') or (name[7]=='3' and name[6]=='4')) and name[14]=='0' and name[15]=='1' and name[16]=='0') or (name[0]=='z' and ((name[6]=='3' and name[7]=='4') or (name[7]=='3' and name[6]=='4')) and name[14]=='0' and name[15]=='1' and name[16]=='0') or (name[0]=='d' and ((name[6]=='1' and name[7]=='7') or (name[7]=='1' and name[6]=='7')) and name[14]=='x' and name[15]=='0' and name[16]=='1') or (name[0]=='z' and ((name[6]=='1' and name[7]=='7') or (name[7]=='1' and name[6]=='7')) and name[14]=='x' and name[15]=='0' and name[16]=='1') or (name[0]=='z' and ((name[6]=='2' and name[7]=='3') or (name[7]=='2' and name[6]=='3')) and name[14]=='0' and name[15]=='0' and name[16]=='1'):
                new_arom += 1
            if (name[0]=='s' and ((name[9]=='1' and name[10]=='2') or (name[10]=='1' and name[9]=='2')) and name[18]=='x' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='1' and name[10]=='7') or (name[10]=='1' and name[9]=='7')) and name[18]=='x' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='2' and name[10]=='3') or (name[10]=='2' and name[9]=='3')) and name[18]=='0' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='3' and name[10]=='4') or (name[10]=='3' and name[9]=='4')) and name[18]=='0' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='4' and name[10]=='5') or (name[10]=='4' and name[9]=='5')) and name[18]=='0' and name[19]=='x' and name[20]=='0') or (name[0]=='s' and ((name[9]=='6' and name[10]=='7') or (name[10]=='6' and name[9]=='7')) and name[18]=='0' and name[19]=='0' and name[20]=='x') or (name[0]=='d' and ((name[9]=='3' and name[10]=='4') or (name[10]=='3' and name[9]=='4')) and name[18]=='0' and name[19]=='1' and name[20]=='0') or (name[0]=='z' and ((name[9]=='3' and name[10]=='4') or (name[10]=='3' and name[9]=='4')) and name[18]=='0' and name[19]=='1' and name[20]=='0') or (name[0]=='d' and ((name[9]=='1' and name[10]=='7') or (name[10]=='1' and name[9]=='7')) and name[18]=='x' and name[19]=='0' and name[20]=='1') or (name[0]=='z' and ((name[9]=='1' and name[10]=='7') or (name[10]=='1' and name[9]=='7')) and name[18]=='x' and name[19]=='0' and name[20]=='1') or (name[0]=='z' and ((name[9]=='2' and name[10]=='3') or (name[10]=='2' and name[9]=='3')) and name[18]=='0' and name[19]=='0' and name[20]=='1'):
                new_arom += 1
        new_aromaticity.append(new_arom)
        if name[0]=='s':
            newdescriptor.append('0')
        elif name[0]=='d':
            newdescriptor.append('1')
        elif name[0]=='z':
            newdescriptor.append('2')
    oxdesc.append(name[-9])
    if ((name[-1]=='0' or name[-1]=='x') and (name[-2]=='0' or name[-2]=='x') and (name[-3]=='0' or name[-3]=='x')) and ((name[-5]=='0' or name[-5]=='x') and (name[-6]=='0' or name[-6]=='x') and (name[-7]=='0' or name[-7]=='x')):
        aromaticity.append('2')
    elif ((name[-1]=='0' or name[-1]=='x') and (name[-2]=='0' or name[-2]=='x') and (name[-3]=='0' or name[-3]=='x')) or ((name[-5]=='0' or name[-5]=='x') and (name[-6]=='0' or name[-6]=='x') and (name[-7]=='0' or name[-7]=='x')):
        aromaticity.append('1')
    else:
        aromaticity.append('0')
    rconn = []
    if 'lin' in name:
        rconn.append(name[8:10])
        rox[pi][0] = name[13]
        rox[pi][1] = name[14]
        rox[pi][2] = name[15]
        rox[pi][3] = name[17]
        rox[pi][4] = name[18]
        rox[pi][5] = name[19]
        alfa = str(name[6])
        if cis in alfa:
           rbond.append('1')
        elif trans in alfa:
           rbond.append('2')
    elif 'cyc' in name:
        rbond.append('0')
        if name[6] > name[9]:
            rconn.append(name[9]+name[6])
        elif name[6] < name[9]:
            rconn.append(name[6]+name[9])
        elif name[6] == name[9]:
            rconn.append(name[6]+name[9])
        if name[7] > name[10]:
            rconn.append(name[10]+name[7])
        elif name[7] < name[10]:
            rconn.append(name[7]+name[10])
        elif name[7] == name[10]:
            rconn.append(name[7]+name[10])
        rox[pi][0] = name[14]
        rox[pi][1] = name[15]
        rox[pi][2] = name[16]
        rox[pi][3] = name[18]
        rox[pi][4] = name[19]
        rox[pi][5] = name[20]
    for k in range(len(rconn)):
        iflag = 0
        for j in range(len(conn)):
            if rconn[k] == conn[j]:
                iflag = 1
                break
        if iflag == 0:
            conn.append(rconn[k])
conn.sort()
for c in range(830):
    for d in range(6):
        if rox[c][d] == 'x':
           rox[c][d] = 0
molec = [[0 for x in range(len(conn))] for y in range(nl)]
infile.seek(0)
nl1 = -1
linnum = 0
beta = 0
excited2 = []
excited3 = []
osc1 = []
osc2 = []
osc3 = []
nom = [0 for x in range(nl)]
energia = [0 for z in range(nl)]
excitat = [0 for t in range(nl)]
for line in infile:
    linnum += 1
    if linnum==1:
        continue
    nl1 += 1
    name = line.rstrip().split()[0]
    energy = line.rstrip().split()[10]
    excited = line.rstrip().split()[3]
    excited2.append(line.rstrip().split()[5])
    excited3.append(line.rstrip().split()[7])
    osc1.append(line.rstrip().split()[4])
    osc2.append(line.rstrip().split()[6])
    osc3.append(line.rstrip().split()[8])
    nom[beta] = name
    energia[beta] = energy
    excitat[beta] = excited
    rconn = []
    if 'lin' in name:
        rconn.append(name[8:10])
    elif 'cyc' in name:
        if name[6] > name[9]:
            rconn.append(name[9]+name[6])
        elif name[6] < name[9]:
            rconn.append(name[6]+name[9])
        elif name[6] == name[9]:
            rconn.append(name[6]+name[9])
        if name[7] > name[10]:
            rconn.append(name[10]+name[7])
        elif name[7] < name[10]:
            rconn.append(name[7]+name[10])
        elif name[7] == name[10]:
            rconn.append(name[7]+name[10])
    for k in range(len(rconn)):
        for j in range(len(conn)):
            if rconn[k] == conn[j]:
                if 'lin' in name:
                        molec[nl1][j] = 1
                elif 'cyc' in name:
                    if k == 0:
                        if name[6] > name[9]:
                            molec[nl1][j] = 2
                        elif name[6] < name[9]:
                            molec[nl1][j] = 1
                        elif name[6] == name[9]:
                            molec[nl1][j] = 1
                    if k == 1:
                        if name[7] > name[10]:
                            molec[nl1][j] = 2
                        elif name[7] < name[10]:
                            molec[nl1][j] = 1
                        elif name[7] == name[10]:
                            molec[nl1][j] = 1
                    if rconn[0] == rconn[1]:
                        molec[nl1][j] = 3
                        break
                        break
    beta += 1
infile.close()
outfile = open('QBB.out','w')
for k in range(nl):
    if k == 0:
        outfile.write('%s\t' % (''.join(str('Molec_name'))))
        for p in range(len(conn)):
            outfile.write('%s_%s\t' % (''.join(str('Conn')),(''.join(str(conn[p])))))
        for t in range(6):
            position = t+1
            outfile.write('%s_%s\t' % (''.join(str('Ox_position')),(''.join(str(position)))))
        outfile.write('%s\t' % (''.join(str('Central_bond'))))
        outfile.write('%s\t' % (''.join(str('LS'))))
        outfile.write('%s\t' % (''.join(str('Ox_state'))))
        outfile.write('%s\t' % (''.join(str('IN'))))
        outfile.write('%s\t' % (''.join(str('AR'))))
        outfile.write('%s\t' % (''.join(str('G_rel'))))
        outfile.write('%s\t' % (''.join(str('E_S1'))))
        outfile.write('%s\t' % (''.join(str('f_S1'))))
        outfile.write('%s\t' % (''.join(str('E_S2'))))
        outfile.write('%s\t' % (''.join(str('f_S2'))))
        outfile.write('%s\t' % (''.join(str('E_S3'))))
        outfile.write('%s\n' % (''.join(str('f_S3'))))
    else:
        hola = k-1
        outfile.write('%s\t' % (''.join(str(nom[hola]))))
        outfile.write('%s\t' % ('\t'.join(str(molec[hola][j]) for j in range(len(conn)))))
        outfile.write('%s\t' % ('\t'.join(str(rox[hola][d]) for d in range(6))))
        outfile.write('%s\t' % (''.join(str(rbond[hola]))))
        outfile.write('%s\t' % (''.join(str(newdescriptor[hola]))))
        outfile.write('%s\t' % (''.join(str(oxdesc[hola]))))
        outfile.write('%s\t' % (''.join(str(aromaticity[hola]))))
        outfile.write('%s\t' % (''.join(str(new_aromaticity[hola]))))
        outfile.write('%s\t' % (''.join(str(energia[hola]))))
        outfile.write('%s\t' % (''.join(str(excitat[hola]))))
        outfile.write('%s\t' % (''.join(str(osc1[hola]))))
        outfile.write('%s\t' % (''.join(str(excited2[hola]))))
        outfile.write('%s\t' % (''.join(str(osc2[hola]))))
        outfile.write('%s\t' % (''.join(str(excited3[hola]))))
        outfile.write('%s\n' % (''.join(str(osc3[hola]))))
outfile.close()

# generate input layer for QBF model

infile = open('dhi_dimer_data.dat','r')
conn = []
rox = [[0 for a in range(6)] for b in range(830)]
rbond = []
newdescriptor = []
oxdesc = []
aromaticity = []
nl = 0
pi = -1
cis = str('c')
trans = str('t')
new_aromaticity = []
listname = []
acd2 = []
acd3 = []
aromcycdim= open('cyc.ox1.ox2.out','r')
for line in aromcycdim:
    listname.append(line.rstrip().split()[0])
    acd2.append(line.rstrip().split()[1])
    acd3.append(line.rstrip().split()[2])
aromcycdim.seek(0)
for line in infile:
    nl += 1
    if nl==1:
        continue
    pi += 1
    name = line.rstrip().split()[0]
    new_arom = 0
    num = 0
    found = 0
    for el in listname:
        if name==el:
            found = 1
            if acd3[num]=='s':
                newdescriptor.append('0')
            elif acd3[num]=='d':
                newdescriptor.append('1')
            elif acd3[num]=='z':
                newdescriptor.append('2')
            if acd2[num]=='NN':
                new_aromaticity.append('0')
            elif acd2[num]=='AN' or acd2[num]=='NA':
                new_aromaticity.append('1')
            elif acd2[num]=='AA':
                new_aromaticity.append('2')
        num += 1
    if found==0:
        if 'lin' in name:
            if (name[0]=='s' and name[8]=='2' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='s' and name[8]=='3' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='s' and name[8]=='4' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='s' and name[8]=='7' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='d' and name[8]=='3' and name[13]=='1' and name[14]=='0' and name[15]=='0') or (name[0]=='d' and name[8]=='4' and name[13]=='0' and name[14]=='1' and name[15]=='0') or (name[0]=='d' and name[8]=='7' and name[13]=='0' and name[14]=='0' and name[15]=='1') or (name[0]=='z' and name[8]=='3' and name[13]=='0' and name[14]=='1' and name[15]=='0') or (name[0]=='z' and name[8]=='3' and name[13]=='0' and name[14]=='0' and name[15]=='1'):
                new_arom += 1
            if (name[0]=='s' and name[9]=='2' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='s' and name[9]=='3' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='s' and name[9]=='4' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='s' and name[9]=='7' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='d' and name[9]=='3' and name[17]=='1' and name[18]=='0' and name[19]=='0') or (name[0]=='d' and name[9]=='4' and name[17]=='0' and name[18]=='1' and name[19]=='0') or (name[0]=='d' and name[9]=='7' and name[17]=='0' and name[18]=='0' and name[19]=='1') or (name[0]=='z' and name[9]=='3' and name[17]=='0' and name[18]=='1' and name[19]=='0') or (name[0]=='z' and name[9]=='3' and name[17]=='0' and name[18]=='0' and name[19]=='1'):
                new_arom += 1
        elif 'cyc' in name:
            if (name[0]=='s' and ((name[6]=='1' and name[7]=='2') or (name[7]=='1' and name[6]=='2')) and name[14]=='x' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='1' and name[7]=='7') or (name[7]=='1' and name[6]=='7')) and name[14]=='x' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='2' and name[7]=='3') or (name[7]=='2' and name[6]=='3')) and name[14]=='0' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='3' and name[7]=='4') or (name[7]=='3' and name[6]=='4')) and name[14]=='0' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='4' and name[7]=='5') or (name[7]=='4' and name[6]=='5')) and name[14]=='0' and name[15]=='x' and name[16]=='0') or (name[0]=='s' and ((name[6]=='6' and name[7]=='7') or (name[7]=='6' and name[6]=='7')) and name[14]=='0' and name[15]=='0' and name[16]=='x') or (name[0]=='d' and ((name[6]=='3' and name[7]=='4') or (name[7]=='3' and name[6]=='4')) and name[14]=='0' and name[15]=='1' and name[16]=='0') or (name[0]=='z' and ((name[6]=='3' and name[7]=='4') or (name[7]=='3' and name[6]=='4')) and name[14]=='0' and name[15]=='1' and name[16]=='0') or (name[0]=='d' and ((name[6]=='1' and name[7]=='7') or (name[7]=='1' and name[6]=='7')) and name[14]=='x' and name[15]=='0' and name[16]=='1') or (name[0]=='z' and ((name[6]=='1' and name[7]=='7') or (name[7]=='1' and name[6]=='7')) and name[14]=='x' and name[15]=='0' and name[16]=='1') or (name[0]=='z' and ((name[6]=='2' and name[7]=='3') or (name[7]=='2' and name[6]=='3')) and name[14]=='0' and name[15]=='0' and name[16]=='1'):
                new_arom += 1
            if (name[0]=='s' and ((name[9]=='1' and name[10]=='2') or (name[10]=='1' and name[9]=='2')) and name[18]=='x' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='1' and name[10]=='7') or (name[10]=='1' and name[9]=='7')) and name[18]=='x' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='2' and name[10]=='3') or (name[10]=='2' and name[9]=='3')) and name[18]=='0' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='3' and name[10]=='4') or (name[10]=='3' and name[9]=='4')) and name[18]=='0' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='4' and name[10]=='5') or (name[10]=='4' and name[9]=='5')) and name[18]=='0' and name[19]=='x' and name[20]=='0') or (name[0]=='s' and ((name[9]=='6' and name[10]=='7') or (name[10]=='6' and name[9]=='7')) and name[18]=='0' and name[19]=='0' and name[20]=='x') or (name[0]=='d' and ((name[9]=='3' and name[10]=='4') or (name[10]=='3' and name[9]=='4')) and name[18]=='0' and name[19]=='1' and name[20]=='0') or (name[0]=='z' and ((name[9]=='3' and name[10]=='4') or (name[10]=='3' and name[9]=='4')) and name[18]=='0' and name[19]=='1' and name[20]=='0') or (name[0]=='d' and ((name[9]=='1' and name[10]=='7') or (name[10]=='1' and name[9]=='7')) and name[18]=='x' and name[19]=='0' and name[20]=='1') or (name[0]=='z' and ((name[9]=='1' and name[10]=='7') or (name[10]=='1' and name[9]=='7')) and name[18]=='x' and name[19]=='0' and name[20]=='1') or (name[0]=='z' and ((name[9]=='2' and name[10]=='3') or (name[10]=='2' and name[9]=='3')) and name[18]=='0' and name[19]=='0' and name[20]=='1'):
                new_arom += 1
        new_aromaticity.append(new_arom)
        if name[0]=='s':
            newdescriptor.append('0')
        elif name[0]=='d':
            newdescriptor.append('1')
        elif name[0]=='z':
            newdescriptor.append('2')
    oxdesc.append(name[-9])
    if ((name[-1]=='0' or name[-1]=='x') and (name[-2]=='0' or name[-2]=='x') and (name[-3]=='0' or name[-3]=='x')) and ((name[-5]=='0' or name[-5]=='x') and (name[-6]=='0' or name[-6]=='x') and (name[-7]=='0' or name[-7]=='x')):
        aromaticity.append('2')
    elif ((name[-1]=='0' or name[-1]=='x') and (name[-2]=='0' or name[-2]=='x') and (name[-3]=='0' or name[-3]=='x')) or ((name[-5]=='0' or name[-5]=='x') and (name[-6]=='0' or name[-6]=='x') and (name[-7]=='0' or name[-7]=='x')):
        aromaticity.append('1')
    else:
        aromaticity.append('0')
    rconn = []
    if 'lin' in name:
        rconn.append(name[8])
        rconn.append(name[9])
        rox[pi][0] = name[13]
        rox[pi][1] = name[14]
        rox[pi][2] = name[15]
        rox[pi][3] = name[17]
        rox[pi][4] = name[18]
        rox[pi][5] = name[19]
        alfa = str(name[6])
        if cis in alfa:
           rbond.append('0')
        elif trans in alfa:
           rbond.append('1')
    elif 'cyc' in name:
        rconn.append(name[6]+name[7])
        if name[9] > name[10]:
            rbond.append('1')
            rconn.append(name[10]+name[9])
        elif name[9] < name[10]:
            rbond.append('0')
            rconn.append(name[9]+name[10])
        rox[pi][0] = name[14]
        rox[pi][1] = name[15]
        rox[pi][2] = name[16]
        rox[pi][3] = name[18]
        rox[pi][4] = name[19]
        rox[pi][5] = name[20]
    for k in range(len(rconn)):
        iflag = 0
        for j in range(len(conn)):
            if rconn[k] == conn[j]:
                iflag = 1
                break
        if iflag == 0:
            conn.append(rconn[k])
conn.sort()
for c in range(830):
    for d in range(6):
        if rox[c][d] == 'x':
           rox[c][d] = 0
molec = [[0 for x in range(len(conn))] for y in range(nl)]
infile.seek(0)
nl1 = -1
beta = 0
nom = [0 for x in range(nl)]
energia = [0 for z in range(nl)]
excitat = [0 for t in range(nl)]
excited2 = []
excited3 = []
osc1 = []
osc2 = []
osc3 = []
linnum = 0
for line in infile:
    linnum += 1
    if linnum==1:
        continue
    nl1 += 1
    name = line.rstrip().split()[0]
    energy = line.rstrip().split()[10]
    excited = line.rstrip().split()[3]
    excited2.append(line.rstrip().split()[5])
    excited3.append(line.rstrip().split()[7])
    osc1.append(line.rstrip().split()[4])
    osc2.append(line.rstrip().split()[6])
    osc3.append(line.rstrip().split()[8])
    nom[beta] = name
    energia[beta] = energy
    excitat[beta] = excited
    rconn = []
    if 'lin' in name:
        rconn.append(name[8])
        rconn.append(name[9])
    elif 'cyc' in name:
        rconn.append(name[6]+name[7])
        if name[9] > name[10]:
            rconn.append(name[10]+name[9])
        elif name[9] < name[10]:
            rconn.append(name[9]+name[10])
    for k in range(len(rconn)):
        for j in range(len(conn)):
            if rconn[k] == conn[j]:
                molec[nl1][j] = 1
                break
                break
    beta += 1
infile.close()
outfile = open('QBF.out','w')
for k in range(nl):
    if k == 0:
        outfile.write('%s\t' % (''.join(str('Molec_name'))))
        for p in range(len(conn)):
            outfile.write('%s_%s\t' % (''.join(str('Conn')),(''.join(str(conn[p])))))
        for t in range(6):
            position = t+1
            outfile.write('%s_%s\t' % (''.join(str('Ox_position')),(''.join(str(position)))))
        outfile.write('%s\t' % (''.join(str('Central_bond'))))
        outfile.write('%s\t' % (''.join(str('LS'))))
        outfile.write('%s\t' % (''.join(str('Ox_state'))))
        outfile.write('%s\t' % (''.join(str('IN'))))
        outfile.write('%s\t' % (''.join(str('AR'))))
        outfile.write('%s\t' % (''.join(str('G_rel'))))
        outfile.write('%s\t' % (''.join(str('E_S1'))))
        outfile.write('%s\t' % (''.join(str('f_S1'))))
        outfile.write('%s\t' % (''.join(str('E_S2'))))
        outfile.write('%s\t' % (''.join(str('f_S2'))))
        outfile.write('%s\t' % (''.join(str('E_S3'))))
        outfile.write('%s\n' % (''.join(str('f_S3'))))
    else:
        hola = k-1
        outfile.write('%s\t' % (''.join(str(nom[hola]))))
        outfile.write('%s\t' % ('\t'.join(str(molec[hola][j]) for j in range(len(conn)))))
        outfile.write('%s\t' % ('\t'.join(str(rox[hola][d]) for d in range(6))))
        outfile.write('%s\t' % (''.join(str(rbond[hola]))))
        outfile.write('%s\t' % (''.join(str(newdescriptor[hola]))))
        outfile.write('%s\t' % (''.join(str(oxdesc[hola]))))
        outfile.write('%s\t' % (''.join(str(aromaticity[hola]))))
        outfile.write('%s\t' % (''.join(str(new_aromaticity[hola]))))
        outfile.write('%s\t' % (''.join(str(energia[hola]))))
        outfile.write('%s\t' % (''.join(str(excitat[hola]))))
        outfile.write('%s\t' % (''.join(str(osc1[hola]))))
        outfile.write('%s\t' % (''.join(str(excited2[hola]))))
        outfile.write('%s\t' % (''.join(str(osc2[hola]))))
        outfile.write('%s\t' % (''.join(str(excited3[hola]))))
        outfile.write('%s\n' % (''.join(str(osc3[hola]))))
outfile.close()

# generate input layer for QBS model

infile = open('dhi_dimer_data.dat','r')
conn = []
conn2 = []
rox = [[0 for a in range(6)] for b in range(830)]
rbond = []
newdescriptor = []
oxdesc = []
aromaticity = []
nl = 0
pi = -1
cis = str('c')
trans = str('t')
new_aromaticity = []
listname = []
acd2 = []
acd3 = []
aromcycdim= open('cyc.ox1.ox2.out','r')
for line in aromcycdim:
    listname.append(line.rstrip().split()[0])
    acd2.append(line.rstrip().split()[1])
    acd3.append(line.rstrip().split()[2])
aromcycdim.seek(0)
for line in infile:
    nl += 1
    if nl==1:
        continue
    pi += 1
    name = line.rstrip().split()[0]
    new_arom = 0
    num = 0
    found = 0
    for el in listname:
        if name==el:
            found = 1
            if acd3[num]=='s':
                newdescriptor.append('0')
            elif acd3[num]=='d':
                newdescriptor.append('1')
            elif acd3[num]=='z':
                newdescriptor.append('2')
            if acd2[num]=='NN':
                new_aromaticity.append('0')
            elif acd2[num]=='AN' or acd2[num]=='NA':
                new_aromaticity.append('1')
            elif acd2[num]=='AA':
                new_aromaticity.append('2')
        num += 1
    if found==0:
        if 'lin' in name:
            if (name[0]=='s' and name[8]=='2' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='s' and name[8]=='3' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='s' and name[8]=='4' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='s' and name[8]=='7' and name[13]=='0' and name[14]=='0' and name[15]=='0') or (name[0]=='d' and name[8]=='3' and name[13]=='1' and name[14]=='0' and name[15]=='0') or (name[0]=='d' and name[8]=='4' and name[13]=='0' and name[14]=='1' and name[15]=='0') or (name[0]=='d' and name[8]=='7' and name[13]=='0' and name[14]=='0' and name[15]=='1') or (name[0]=='z' and name[8]=='3' and name[13]=='0' and name[14]=='1' and name[15]=='0') or (name[0]=='z' and name[8]=='3' and name[13]=='0' and name[14]=='0' and name[15]=='1'):
                new_arom += 1
            if (name[0]=='s' and name[9]=='2' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='s' and name[9]=='3' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='s' and name[9]=='4' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='s' and name[9]=='7' and name[17]=='0' and name[18]=='0' and name[19]=='0') or (name[0]=='d' and name[9]=='3' and name[17]=='1' and name[18]=='0' and name[19]=='0') or (name[0]=='d' and name[9]=='4' and name[17]=='0' and name[18]=='1' and name[19]=='0') or (name[0]=='d' and name[9]=='7' and name[17]=='0' and name[18]=='0' and name[19]=='1') or (name[0]=='z' and name[9]=='3' and name[17]=='0' and name[18]=='1' and name[19]=='0') or (name[0]=='z' and name[9]=='3' and name[17]=='0' and name[18]=='0' and name[19]=='1'):
                new_arom += 1
        elif 'cyc' in name:
            if (name[0]=='s' and ((name[6]=='1' and name[7]=='2') or (name[7]=='1' and name[6]=='2')) and name[14]=='x' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='1' and name[7]=='7') or (name[7]=='1' and name[6]=='7')) and name[14]=='x' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='2' and name[7]=='3') or (name[7]=='2' and name[6]=='3')) and name[14]=='0' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='3' and name[7]=='4') or (name[7]=='3' and name[6]=='4')) and name[14]=='0' and name[15]=='0' and name[16]=='0') or (name[0]=='s' and ((name[6]=='4' and name[7]=='5') or (name[7]=='4' and name[6]=='5')) and name[14]=='0' and name[15]=='x' and name[16]=='0') or (name[0]=='s' and ((name[6]=='6' and name[7]=='7') or (name[7]=='6' and name[6]=='7')) and name[14]=='0' and name[15]=='0' and name[16]=='x') or (name[0]=='d' and ((name[6]=='3' and name[7]=='4') or (name[7]=='3' and name[6]=='4')) and name[14]=='0' and name[15]=='1' and name[16]=='0') or (name[0]=='z' and ((name[6]=='3' and name[7]=='4') or (name[7]=='3' and name[6]=='4')) and name[14]=='0' and name[15]=='1' and name[16]=='0') or (name[0]=='d' and ((name[6]=='1' and name[7]=='7') or (name[7]=='1' and name[6]=='7')) and name[14]=='x' and name[15]=='0' and name[16]=='1') or (name[0]=='z' and ((name[6]=='1' and name[7]=='7') or (name[7]=='1' and name[6]=='7')) and name[14]=='x' and name[15]=='0' and name[16]=='1') or (name[0]=='z' and ((name[6]=='2' and name[7]=='3') or (name[7]=='2' and name[6]=='3')) and name[14]=='0' and name[15]=='0' and name[16]=='1'):
                new_arom += 1
            if (name[0]=='s' and ((name[9]=='1' and name[10]=='2') or (name[10]=='1' and name[9]=='2')) and name[18]=='x' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='1' and name[10]=='7') or (name[10]=='1' and name[9]=='7')) and name[18]=='x' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='2' and name[10]=='3') or (name[10]=='2' and name[9]=='3')) and name[18]=='0' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='3' and name[10]=='4') or (name[10]=='3' and name[9]=='4')) and name[18]=='0' and name[19]=='0' and name[20]=='0') or (name[0]=='s' and ((name[9]=='4' and name[10]=='5') or (name[10]=='4' and name[9]=='5')) and name[18]=='0' and name[19]=='x' and name[20]=='0') or (name[0]=='s' and ((name[9]=='6' and name[10]=='7') or (name[10]=='6' and name[9]=='7')) and name[18]=='0' and name[19]=='0' and name[20]=='x') or (name[0]=='d' and ((name[9]=='3' and name[10]=='4') or (name[10]=='3' and name[9]=='4')) and name[18]=='0' and name[19]=='1' and name[20]=='0') or (name[0]=='z' and ((name[9]=='3' and name[10]=='4') or (name[10]=='3' and name[9]=='4')) and name[18]=='0' and name[19]=='1' and name[20]=='0') or (name[0]=='d' and ((name[9]=='1' and name[10]=='7') or (name[10]=='1' and name[9]=='7')) and name[18]=='x' and name[19]=='0' and name[20]=='1') or (name[0]=='z' and ((name[9]=='1' and name[10]=='7') or (name[10]=='1' and name[9]=='7')) and name[18]=='x' and name[19]=='0' and name[20]=='1') or (name[0]=='z' and ((name[9]=='2' and name[10]=='3') or (name[10]=='2' and name[9]=='3')) and name[18]=='0' and name[19]=='0' and name[20]=='1'):
                new_arom += 1
        new_aromaticity.append(new_arom)
        if name[0]=='s':
            newdescriptor.append('0')
        elif name[0]=='d':
            newdescriptor.append('1')
        elif name[0]=='z':
            newdescriptor.append('2')
    oxdesc.append(name[-9])
    if ((name[-1]=='0' or name[-1]=='x') and (name[-2]=='0' or name[-2]=='x') and (name[-3]=='0' or name[-3]=='x')) and ((name[-5]=='0' or name[-5]=='x') and (name[-6]=='0' or name[-6]=='x') and (name[-7]=='0' or name[-7]=='x')):
        aromaticity.append('2')
    elif ((name[-1]=='0' or name[-1]=='x') and (name[-2]=='0' or name[-2]=='x') and (name[-3]=='0' or name[-3]=='x')) or ((name[-5]=='0' or name[-5]=='x') and (name[-6]=='0' or name[-6]=='x') and (name[-7]=='0' or name[-7]=='x')):
        aromaticity.append('1')
    else:
        aromaticity.append('0')
    rconn = []
    rconn2 = []
    if 'lin' in name:
        rconn.append(name[8])
        rconn2.append(name[9])
        rox[pi][0] = name[13]
        rox[pi][1] = name[14]
        rox[pi][2] = name[15]
        rox[pi][3] = name[17]
        rox[pi][4] = name[18]
        rox[pi][5] = name[19]
        alfa = str(name[6])
        if cis in alfa:
           rbond.append('0')
        elif trans in alfa:
           rbond.append('1')
    elif 'cyc' in name:
        rconn.append(name[6])
        rconn.append(name[7])
        rconn2.append(name[9])
        rconn2.append(name[10])
        if name[9] > name[10]:
            rbond.append('1')
        elif name[9] < name[10]:
            rbond.append('0')
        rox[pi][0] = name[14]
        rox[pi][1] = name[15]
        rox[pi][2] = name[16]
        rox[pi][3] = name[18]
        rox[pi][4] = name[19]
        rox[pi][5] = name[20]
    for k in range(len(rconn)):
        iflag = 0
        for j in range(len(conn)):
            if rconn[k] == conn[j]:
                iflag = 1
                break
        if iflag == 0:
            conn.append(rconn[k])
    for k2 in range(len(rconn2)):
        iflag = 0
        for j2 in range(len(conn2)):
            if rconn2[k2] == conn2[j2]:
                iflag = 1
                break
        if iflag == 0:
            conn2.append(rconn2[k2])
conn.sort()
conn2.sort()
for c in range(830):
    for d in range(6):
        if rox[c][d] == 'x':
           rox[c][d] = 0
molec = [[0 for x in range(len(conn))] for y in range(nl)]
molec2 = [[0 for x in range(len(conn2))] for y in range(nl)]
infile.seek(0)
nl1 = -1
beta = 0
nom = [0 for x in range(nl)]
energia = [0 for z in range(nl)]
excitat = [0 for t in range(nl)]
excited2 = []
excited3 = []
osc1 = []
osc2 = []
osc3 = []
linnum = 0
for line in infile:
    linnum += 1
    if linnum==1:
        continue
    nl1 += 1
    name = line.rstrip().split()[0]
    energy = line.rstrip().split()[10]
    excited = line.rstrip().split()[3]
    excited2.append(line.rstrip().split()[5])
    excited3.append(line.rstrip().split()[7])
    osc1.append(line.rstrip().split()[4])
    osc2.append(line.rstrip().split()[6])
    osc3.append(line.rstrip().split()[8])
    nom[beta] = name
    energia[beta] = energy
    excitat[beta] = excited
    rconn = []
    rconn2 = []
    if 'lin' in name:
        rconn.append(name[8])
        rconn2.append(name[9])
    elif 'cyc' in name:
        rconn.append(name[6])
        rconn.append(name[7])
        rconn2.append(name[9])
        rconn2.append(name[10])
    for k in range(len(rconn)):
        for j in range(len(conn)):
            if rconn[k] == conn[j]:
                if 'lin' in name:
                    molec[nl1][j] = 1
                if 'cyc' in name:
                    molec[nl1][j] = 2
                break
                break
    for k2 in range(len(rconn2)):
        for j2 in range(len(conn2)):
            if rconn2[k2] == conn2[j2]:
                if 'lin' in name:
                    molec2[nl1][j2] = 1
                if 'cyc' in name:
                    molec2[nl1][j2] = 2
                break
                break
    beta += 1
infile.close()
outfile = open('QBS.out','w')
for k in range(nl):
    if k == 0:
        outfile.write('%s\t' % (''.join(str('Molec_name'))))
        for p in range(len(conn)):
            outfile.write('%s_%s\t' % (''.join(str('Conn')),(''.join(str(conn[p])))))
        for p2 in range(len(conn2)):
            outfile.write('%s_%s\t' % (''.join(str('Conn2')),(''.join(str(conn2[p2])))))
        for t in range(6):
            position = t+1
            outfile.write('%s_%s\t' % (''.join(str('Ox_position')),(''.join(str(position)))))
        outfile.write('%s\t' % (''.join(str('Central_bond'))))
        outfile.write('%s\t' % (''.join(str('LS'))))
        outfile.write('%s\t' % (''.join(str('Ox_state'))))
        outfile.write('%s\t' % (''.join(str('IN'))))
        outfile.write('%s\t' % (''.join(str('AR'))))
        outfile.write('%s\t' % (''.join(str('G_rel'))))
        outfile.write('%s\t' % (''.join(str('E_S1'))))
        outfile.write('%s\t' % (''.join(str('f_S1'))))
        outfile.write('%s\t' % (''.join(str('E_S2'))))
        outfile.write('%s\t' % (''.join(str('f_S2'))))
        outfile.write('%s\t' % (''.join(str('E_S3'))))
        outfile.write('%s\n' % (''.join(str('f_S3'))))
    else:
        hola = k-1
        outfile.write('%s\t' % (''.join(str(nom[hola]))))
        outfile.write('%s\t' % ('\t'.join(str(molec[hola][j]) for j in range(len(conn)))))
        outfile.write('%s\t' % ('\t'.join(str(molec2[hola][j2]) for j2 in range(len(conn2)))))
        outfile.write('%s\t' % ('\t'.join(str(rox[hola][d]) for d in range(6))))
        outfile.write('%s\t' % (''.join(str(rbond[hola]))))
        outfile.write('%s\t' % (''.join(str(newdescriptor[hola]))))
        outfile.write('%s\t' % (''.join(str(oxdesc[hola]))))
        outfile.write('%s\t' % (''.join(str(aromaticity[hola]))))
        outfile.write('%s\t' % (''.join(str(new_aromaticity[hola]))))
        outfile.write('%s\t' % (''.join(str(energia[hola]))))
        outfile.write('%s\t' % (''.join(str(excitat[hola]))))
        outfile.write('%s\t' % (''.join(str(osc1[hola]))))
        outfile.write('%s\t' % (''.join(str(excited2[hola]))))
        outfile.write('%s\t' % (''.join(str(osc2[hola]))))
        outfile.write('%s\t' % (''.join(str(excited3[hola]))))
        outfile.write('%s\n' % (''.join(str(osc3[hola]))))
outfile.close()

