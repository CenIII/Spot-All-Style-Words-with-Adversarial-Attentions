

class StyleMarker(object):
	"""docstring for StyleMarker"""
	def __init__(self):
		super(StyleMarker, self).__init__()
		
		self.model = None

	def reloadModel(self):
		self.model = None


	def mark(self,text):

		

		return ptList

	def sep(self,text):
		ptList = self.mark(text)
		markerList = []
		brk_text = ''
		pt = 0
		for (st,ed) in ptList:
			markerList.append(text[st:ed])
			brk_text += text[pt:st]
			pt = ed
		brk_text += text[pt:]
		return brk_text, markerList
		