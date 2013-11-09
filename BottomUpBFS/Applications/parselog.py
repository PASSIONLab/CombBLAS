import re
import itertools

infilename = "nondeterminism_fully_exposed"
oufilename = "parsed.log"

s = 'tokenize these words'
words = re.compile(r'\b\w+\b|\$')  # \b: matches the empty string at the beginning and end of a word
baMVA = re.compile(r'\d+')
tokens = words.findall(s)
print tokens

GENLEN = 20
BRALEN = 17

# looking for "End of all"
# list all with parents set
# find intersection of two runs


class MP_BusData:
	bus_i, ide, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin = range(13)
class PS_GenData:
	bus_i, l_id, Pg, Qg, Qmax, Qmin, Vg, ireg, mBase, ZR, ZX, RT, XT, GTAP, status, rmpct, Pmax, Pmin,owner, frac = range(GENLEN)
class PS_BranchData:
	fbus,tbus, ckt, r, x, b,rateA, rateB, rateC,GI,BI,GJ,BJ, status, length, owner, frac = range(BRALEN)

# In MatPower, there is no specific "load data" section. 
# A bus is listed as type 2 iff it can hold its voltage, making it a PV bus
# Pd, Qd are load MV and load MVAR (http://www.ee.washington.edu/research/pstca/formats/cdf.txt)

LLDS = list()	# list of load entries
Owners = dict()	# dictionary of owners
oufile = open(oufilename, "w")
infile = open(infilename, "r")
areas = []
while infile: 
	line = infile.readline()
	if line.startswith('End of all'): 
		busline = infile.readline() # read the next line [this is a huge one]
		
		baseMVA = baMVA.findall(line)	# findall returns a list
		joined = ''.join(baseMVA)
		tofile = "0," + joined + "\n"
		oufile.write(tofile)
	if line.startswith('mpc.bus'):
		oufile.writelines('IEEE 300 Bus system\n')
		oufile.writelines('Converted from MatPower by Aydin Buluc\n')
		busline = infile.readline() # read the next line
		while not(busline.startswith("];")):
			ind = busline.rfind(";");
			busline = busline[:ind]	# remove the end of line ';' and any trailing whitespace
			IL = busline.split() # input list
			# print IL
			comma = ",\t"
			BS = [None]*BUSLEN		# list for one bus entry
			LD = [None]*LOALEN		# list for one load entry 
			for index,item in enumerate(IL):
				if index == MP_BusData.bus_i:
					BS[PS_BusData.bus_i] = item
					BS[PS_BusData.name] = "\'BUS" + item + "\'"
					LD[PS_LoadData.bus_i] = item	# also write to load
				elif index == MP_BusData.ide:		# ide = type of the bus
					BS[PS_BusData.ide] = item
				elif index == MP_BusData.Pd:
					LD[PS_LoadData.Pl] = item 	# belongs to the load data
				elif index == MP_BusData.Qd:
					LD[PS_LoadData.Ql] = item	# belongs to the load data
				elif index == MP_BusData.Gs:
					BS[PS_BusData.Gs] = item 	
				elif index == MP_BusData.Bs:
					BS[PS_BusData.Bs] = item 	
				elif index == MP_BusData.area:		# include area information too
					BS[PS_BusData.area] = item
					LD[PS_LoadData.area] = item	# also write to load	
					areas.append(item)
				elif index == MP_BusData.Vm:
					BS[PS_BusData.Vm] = item 	
				elif index == MP_BusData.Va:
					BS[PS_BusData.Va] = item 	
				elif index == MP_BusData.baseKV:
					BS[PS_BusData.baseKV] = item
				elif index == MP_BusData.zone:
					BS[PS_BusData.zone] = item
					LD[PS_LoadData.zone] = item	# also write to load	
			BS[PS_BusData.owner] = '1'
			# print BS
			if float(LD[PS_LoadData.Pl]) > 0 or BS[PS_BusData.ide] == '1':
				LD[PS_LoadData.l_id] = "\'1\'"
				LD[PS_LoadData.status] = '1'
				LD[PS_LoadData.IP] = '0.0'
				LD[PS_LoadData.IQ] = '0.0'
				LD[PS_LoadData.YP] = '0.0'
				LD[PS_LoadData.YQ] = '0.0'
				LD[PS_LoadData.owner] = '1'
				LLDS.append(LD)
			busjoined = comma.join(BS)	
			oufile.write(busjoined+"\n")
			busline = infile.readline() # read the next line
		oufile.write("0 / END OF BUS DATA, BEGIN LOAD DATA\n")
		for ld in LLDS:
			loadjoined = comma.join(ld)
			oufile.write(loadjoined+"\n")
		oufile.write("0 / END OF LOAD DATA, BEGIN GENERATOR DATA\n")
	if line.startswith('mpc.gen'):
		genline = infile.readline() # read the next line
		while not(genline.startswith("];")):
			ind = genline.rfind(";");
			genline = genline[:ind]	# remove the end of line ';' and any trailing whitespace
			IL = genline.split() # input list
			# print IL
			comma = ",\t"
			GN = [None]*GENLEN		# list for one generator entry
			for index,item in enumerate(IL):
				if index == MP_GenData.bus_i:
					GN[PS_GenData.bus_i] = item
				elif index == MP_GenData.Pg:	
					GN[PS_GenData.Pg] = item
				elif index == MP_GenData.Qg:
					GN[PS_GenData.Qg] = item 
				elif index == MP_GenData.Qmax:		
					GN[PS_GenData.Qmax] = item
				elif index == MP_GenData.Qmin:
					GN[PS_GenData.Qmin] = item 
				elif index == MP_GenData.Vg:
					GN[PS_GenData.Vg] = item 
				elif index == MP_GenData.mBase:
					GN[PS_GenData.mBase] = item 
				elif index == MP_GenData.Pmax:		
					GN[PS_GenData.Pmax] = item
				elif index == MP_GenData.Pmin:
					GN[PS_GenData.Pmin] = item 
				elif index == MP_GenData.status:
					GN[PS_GenData.status] = item 
			GN[PS_GenData.l_id] = "\'1\'" 
			GN[PS_GenData.ireg] = '0'
			GN[PS_GenData.ZR] = '0.0'
			GN[PS_GenData.ZX] = '1.0'
			GN[PS_GenData.RT] = '0.0'
			GN[PS_GenData.XT] = '0.0'
			GN[PS_GenData.GTAP] = '1.0'
			GN[PS_GenData.rmpct] = '100.0'
			GN[PS_GenData.owner] = '1'	# everyone is owned by '1'
			GN[PS_GenData.frac] = '1.0'	# at full fraction

			genjoined = comma.join(GN)	
			oufile.write(genjoined+"\n")
			genline = infile.readline() # read the next line
		oufile.write("0 / END OF GENERATOR DATA, BEGIN BRANCH DATA\n")
	if line.startswith('mpc.branch'):
		brline = infile.readline() # read the next line
		while not(brline.startswith("];")):
			ind = brline.rfind(";");
			brline = brline[:ind]	# remove the end of line ';' and any trailing whitespace
			IL = brline.split() # input list
			# print IL
			comma = ",\t"
			BR = [None]*BRALEN		# list for one branch entry
			for index,item in enumerate(IL):
				if index == MP_BranchData.fbus:
					fbus = int(item);
				elif index == MP_BranchData.tbus:
					tbus = int(item)	
				elif index == MP_BranchData.r:
					BR[PS_BranchData.r] = item 
				elif index == MP_BranchData.x:		
					BR[PS_BranchData.x] = item
				elif index == MP_BranchData.b:
					BR[PS_BranchData.b] = item 
				elif index == MP_BranchData.rateA:
					BR[PS_BranchData.rateA] = item 
				elif index == MP_BranchData.rateB:
					BR[PS_BranchData.rateB] = item 
				elif index == MP_BranchData.rateC:		
					BR[PS_BranchData.rateC] = item
				elif index == MP_BranchData.status:
					BR[PS_BranchData.status] = item 
			BR[PS_BranchData.fbus] = str(min(fbus, tbus))	# my PSS/E reader assumes branches are from low_id to high_id
			BR[PS_BranchData.tbus] = '-' + str(max(fbus, tbus))		# to_bus is prefix with minus sign in PSS/E
			BR[PS_BranchData.ckt] = "\'1\'" 
			BR[PS_BranchData.GI] = '0'
			BR[PS_BranchData.BI] = '0'
			BR[PS_BranchData.GJ] = '0'
			BR[PS_BranchData.BJ] = '0'
			BR[PS_BranchData.length] = '0.0'
			BR[PS_BranchData.owner] = '1'	# everyone is owned by '1'
			BR[PS_BranchData.frac] = '1.0'	# at full fraction

			brjoined = comma.join(BR)	
			oufile.write(brjoined+"\n")
			brline = infile.readline() # read the next line
		oufile.write("0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA\n")
		oufile.write("0 / END OF TRANSFORMER DATA, BEGIN AREA DATA\n")
		setone = set(areas)
		for n in setone:
			areaname = 'AREA_'+str(n)
			oufile.write(str(n)+ ", 1, 0.0, 10.0," + areaname + ",\n")
		oufile.write("0 / END OF AREA DATA, BEGIN TWO-TERMINAL DC DATA\n")
		oufile.write("0 / END OF TWO-TERMINAL DC DATA, BEGIN VSC DC LINE DATA\n")
		oufile.write("0 / END OF VSC DC LINE DATA, BEGIN SWITCHED SHUNT DATA\n")
		oufile.write("0 / END OF MULTI-SECTION LINE DATA, BEGIN ZONE DATA\n")
		oufile.write("0 / END OF ZONE DATA, BEGIN INTER-AREA TRANSFER DATA\n")
		oufile.write("0 / END OF INTER-AREA TRANSFER DATA, BEGIN OWNER DATA\n")
		oufile.write("0 / END OF OWNER DATA, BEGIN FACTS DEVICE DATA\n")
		oufile.write("0 / END OF FACTS DEVICE DATA\n")
		exit()
