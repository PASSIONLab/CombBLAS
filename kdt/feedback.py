import os
import getpass
import email
import smtplib
import socket

class feedback:

	_kdt_LogFname = '.KDT_log';
	_kdt_EmailFname = 'KDT_email';
	
	#NOTE:  as of 2011Jan17, putting multiple email addresses in the _kdt_Alias
	#	variable, whether separated by spaces, commas, or semicolons, did
	#	not work properly.
	_kdt_Alias = 'kdt-suggestions@lists.sourceforge.net'
	_kdt_Nlines2Send = 30;
	
	
	@staticmethod
	def startFeedback():
		try:
			import IPython.ipapi                     
		except ImportError:
			try:
				import IPython.core.ipapi                           
			except ImportError:
				feedback.IPYTHON_ACTIVE = False;
				return
			else:                                   
				ip = IPython.core.ipapi.get()
		else:
			ip = IPython.ipapi.get()
		
		if ip == None:
			feedback.IPYTHON_ACTIVE = False;
			return;
		feedback.IPYTHON_ACTIVE = True;
		ip.magic('logstart %s over' % feedback._kdt_LogFname);
	
	
	@staticmethod
	def sendFeedback(nlines=_kdt_Nlines2Send,addr=_kdt_Alias):
		"""
		sends feedback to KDT developers, consisting of the last several lines
		of input from the user (possibly edited by an external program).
	
		sendFeedback only works when IPython is the Python language processor
		used to execute KDT.  It places the last several lines of user input,
		places them into a temporary file, gives the user the chance to edit
		the file to add any other relevant information, then emails the file
		to the KDT feedback alias.
	
		Input Arguments:  
			nlines:  an optional argument denoting the number of prior 
			    lines of input to send, default 30.
			addr:  an optinal argument denoting the email address to which 
			    to send feedback, default
			    kdt-suggestions@lists.sourceforge.net.
		"""
		if feedback.IPYTHON_ACTIVE:
			if not os.path.exists(feedback._kdt_LogFname) or not os.path.isfile(feedback._kdt_LogFname):
				print "logging apparently not enabled for KDT"
				#return;
			
			logFid = open(feedback._kdt_LogFname);
			count = 0;
			# count the lines in the file
			while 1:
				line = logFid.readline()
				if not line:
					break;
				count += 1;
			logFid.close();
			logFid = open(feedback._kdt_LogFname);
			# space past all but last N lines
			for i in range(count - nlines):   
				logFid.readline();
			# copy last N lines to new file
			emailFid = open(feedback._kdt_EmailFname,'w+');
			for i in range(nlines):
				line = logFid.readline();
				emailFid.writelines(line);
			logFid.close();
			emailFid.close();
				
			str = "The code example you want to send to the KDT developers is \nin %s/%s.  \nIf you wish, edit it with your favorite editor.  Type 'Send' \nwhen you are ready to send it or 'Cancel' to cancel sending it.\n>>>" % (os.getcwd(),feedback._kdt_EmailFname)
				
			resp = raw_input(str);
			if resp == 'Send' or resp == 'send':
				#print "Emailing the file."
				msg = email.message_from_file(open(feedback._kdt_EmailFname));
				userAddress = '%s@%s' % (getpass.getuser(), socket.gethostname());
				if os.name is 'posix':
					import pwd
					pwdEntry = pwd.getpwuid(os.getuid());
					userFullName = pwdEntry[4];
					Subject = 'KDT feedback from %s' % pwdEntry[4]
				else:
					Subject = 'KDT feedback from %s' % userAddress
				msg['Subject'] = Subject
				msg['From'] = userAddress;
				msg['To'] = addr;
				try:
					s = smtplib.SMTP('localhost');
					s.sendmail(userAddress, [addr], msg.as_string());
					s.quit();
				except:
					print "Failed to connect via localhost. Please manually email file %s to %s" % (feedback._kdt_EmailFname, addr)
			else:
				print "Canceling the send."
	
	
feedback.startFeedback();
sendFeedback = feedback.sendFeedback
