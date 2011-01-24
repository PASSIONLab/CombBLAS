import os
import pwd
import email
import smtplib
import socket
import IPython.ipapi;

class feedback:

	_kdt_LogFname = '.KDT_log';
	_kdt_EmailFname = 'KDT_email';
	
	#FIX:  replace the following line by the alias, when the alias exists
	#NOTE:  as of 2011Jan17, putting multiple email addresses in the _kdt_Alias
	#	variable, whether separated by spaces, commas, or semicolons, did
	#	not work properly.
	_kdt_Alias = 'kdt-suggestions@lists.sourceforge.net'
	_kdt_Nlines2Send = 30;
	
	
	@staticmethod
	def startFeedback():
		ip = IPython.ipapi.get()
		if ip == None:
			feedback.IPYTHON_ACTIVE = False;
			return;
		feedback.IPYTHON_ACTIVE = True;
		ip.magic('logstart %s over' % feedback._kdt_LogFname);
	
	
	@staticmethod
	def sendFeedback(nlines=_kdt_Nlines2Send,addr=_kdt_Alias):
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
				pwdEntry = pwd.getpwuid(os.getuid());
				userAddress = '%s@%s' % (pwdEntry[0], socket.gethostname());
				userFullName = pwdEntry[4];
				msg['Subject'] = 'KDT feedback from %s' % userFullName;
				msg['From'] = userAddress;
				msg['To'] = addr;
				s = smtplib.SMTP('localhost');
				s.sendmail(userAddress, [addr], msg.as_string());
				s.quit();
			else:
				print "Canceling the send."
	
	
feedback.startFeedback();
