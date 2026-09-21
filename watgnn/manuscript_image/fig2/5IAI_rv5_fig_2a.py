import cPickle, base64
try:
	from SimpleSession.versions.v65 import beginRestore,\
	    registerAfterModelsCB, reportRestoreError, checkVersion
except ImportError:
	from chimera import UserError
	raise UserError('Cannot open session that was saved in a'
	    ' newer version of Chimera; update your version')
checkVersion([1, 19, 42556])
import chimera
from chimera import replyobj
replyobj.status('Restoring session...', \
    blankAfter=0)
replyobj.status('Beginning session restore...', \
    blankAfter=0, secondary=True)
beginRestore()

def restoreCoreModels():
	from SimpleSession.versions.v65 import init, restoreViewer, \
	     restoreMolecules, restoreColors, restoreSurfaces, \
	     restoreVRML, restorePseudoBondGroups, restoreModelAssociations
	molInfo = cPickle.loads(base64.b64decode('gAJ9cQEoVRFyaWJib25JbnNpZGVDb2xvcnECSwJOfYdVCWJhbGxTY2FsZXEDSwJHP9AAAAAAAAB9h1UJcG9pbnRTaXplcQRLAkc/8AAAAAAAAH2HVQVjb2xvcnEFSwJLAH1xBksBXXEHSwFhc4dVCnJpYmJvblR5cGVxCEsCSwB9h1UKc3RpY2tTY2FsZXEJSwJHP/AAAAAAAAB9h1UMbW1DSUZIZWFkZXJzcQpdcQsoTk5lVQxhcm9tYXRpY01vZGVxDEsCSwF9h1UKdmR3RGVuc2l0eXENSwJHQBQAAAAAAAB9h1UGaGlkZGVucQ5LAol9h1UNYXJvbWF0aWNDb2xvcnEPSwJOfYdVD3JpYmJvblNtb290aGluZ3EQSwJLAH2HVQlhdXRvY2hhaW5xEUsCiH2HVQpwZGJWZXJzaW9ucRJLAksCfYdVCG9wdGlvbmFscRN9cRRVCG9wZW5lZEFzcRWIiUsCKFg5AAAAQzpcVXNlcnNcdXNlclxEZXNrdG9wXFdhdEdOTl9hbGxcV2F0R05OLW1haW5cd2F0Z25uXHgucGRiVQNQREJOiXRxFn1xFyhYQwAAAEM6XFVzZXJzXHVzZXJcRGVza3RvcFxXYXRHTk5fYWxsXFdhdEdOTi1tYWluXHdhdGdublx4LnBkYl9wcm9iZS5wZGJVA1BEQk6JdHEYXXEZSwFhc4eHc1UPbG93ZXJDYXNlQ2hhaW5zcRpLAol9h1UJbGluZVdpZHRocRtLAkc/8AAAAAAAAH2HVQ9yZXNpZHVlTGFiZWxQb3NxHEsCSwB9h1UEbmFtZXEdSwJYDwAAAHgucGRiX3Byb2JlLnBkYn1xHlgFAAAAeC5wZGJdcR9LAGFzh1UPYXJvbWF0aWNEaXNwbGF5cSBLAol9h1UPcmliYm9uU3RpZmZuZXNzcSFLAkc/6ZmZmZmZmn2HVQpwZGJIZWFkZXJzcSJdcSMofXEkfXElZVUDaWRzcSZLAksASwCGfXEnSwRLAIZdcShLAWFzh1UOc3VyZmFjZU9wYWNpdHlxKUsCR7/wAAAAAAAAfYdVEGFyb21hdGljTGluZVR5cGVxKksCSwJ9h1UUcmliYm9uSGlkZXNNYWluY2hhaW5xK0sCiH2HVQdkaXNwbGF5cSxLAol9cS2IXXEuSwBhc4d1Lg=='))
	resInfo = cPickle.loads(base64.b64decode('gAJ9cQEoVQZpbnNlcnRxAksNVQEgfYdVC2ZpbGxEaXNwbGF5cQNLDYl9h1UEbmFtZXEESw1YAwAAAFBSQn1xBShYAwAAAENOVF1xBksDYVgDAAAARVhCXXEHSwxhWAMAAABWQUxdcQhLAGFYAwAAAFBCU11xCUsCYVgDAAAAQUxBXXEKSwFhWAMAAABFWFRdcQsoSwVLB0sJSwtldYdVBWNoYWlucQxLDVgBAAAAQX1xDShYAQAAAENOXXEOSwNLAYZxD2GGWAEAAABCTl1xEEsCSwGGcRFhhnWHVQ5yaWJib25EcmF3TW9kZXESSw1LAn2HVQJzc3ETSw2JiYZ9h1UIbW9sZWN1bGVxFEsNSwF9cRVLAE5dcRZLAEsEhnEXYYZzh1ULcmliYm9uQ29sb3JxGEsNSwR9cRkoSwJOXXEaSwJLAYZxG2GGSwNOXXEcSwNLAYZxHWGGTk5dcR5LAEsChnEfYYZLBk5dcSAoSwlLAYZxIUsLSwKGcSJlhksFTl1xIyhLBUsBhnEkSwdLAYZxJWWGdYdVBWxhYmVscSZLDVgAAAAAfYdVCmxhYmVsQ29sb3JxJ0sNSwR9cSgoSwJOXXEpSwJLAYZxKmGGSwNOXXErSwNLAYZxLGGGTk5dcS1LAEsChnEuYYZLBk5dcS8oSwlLAYZxMEsLSwKGcTFlhksFTl1xMihLBUsBhnEzSwdLAYZxNGWGdYdVCGZpbGxNb2RlcTVLDUsBfYdVBWlzSGV0cTZLDYh9cTeJTl1xOEsASwKGcTlhhnOHVQtsYWJlbE9mZnNldHE6Sw1OfYdVCHBvc2l0aW9ucTtdcTwoS6RLAoZxPU2NBEsBhnE+TXUISwGGcT9LpUsBhnFAS6VLAYZxQUulSwGGcUJLpUsBhnFDS6VLAYZxREulSwGGcUVLpUsBhnFGS6VLAYZxR0ulSwGGcUhlVQ1yaWJib25EaXNwbGF5cUlLDYl9h1UIb3B0aW9uYWxxSn1VBHNzSWRxS0sNSv////99h3Uu'))
	atomInfo = cPickle.loads(base64.b64decode('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'))
	bondInfo = cPickle.loads(base64.b64decode('gAJ9cQEoVQVjb2xvcnECSwtOfYdVBWF0b21zcQNdcQQoXXEFKEsPSxBlXXEGKEsQSxNlXXEHKEsQSxFlXXEIKEsRSxJlXXEJKEsTSxVlXXEKKEsTSxRlXXELKEsWSxdlXXEMKEsXSxhlXXENKEsXSxplXXEOKEsYSxllXXEPKEsRSxZlZVUFbGFiZWxxEEsLWAAAAAB9h1UIaGFsZmJvbmRxEUsLiH2HVQZyYWRpdXNxEksLRz/JmZmgAAAAfYdVC2xhYmVsT2Zmc2V0cRNLC059h1UIZHJhd01vZGVxFEsLSwF9h1UIb3B0aW9uYWxxFX1VB2Rpc3BsYXlxFksLSwJ9h3Uu'))
	crdInfo = cPickle.loads(base64.b64decode('gAJ9cQEoSwB9cQIoSwBdcQMoR0BHbfO2RaHLR0ARsCDEm6XjR0BDTU/fO2Rah3EER0BICwIMSbpeR0AUybpeNT99R0BDQk3S8an8h3EFR0BIi2RaHKwIR0ASZmZmZmZmR0BDvrhR64Ufh3EGR0BIXO2RaHKwR0APtkWhysCDR0BEPhR64Ueuh3EHR0BH6XjU/fO2R0Aa0OVgQYk3R0BDY/fO2RaHh3EIR0BHqRaHKwIMR0Abk3S8an76R0BEGwIMSbpeh3EJR0BIhT987ZFoR0AeWBBiTdLyR0BDN64UeuFIh3EKR0BJLbItDlYER0ATEGJN0vGqR0BDkzMzMzMzh3ELR0BJvKwIMSbpR0ART987ZFodR0BD/KwIMSbph3EMR0BKVqfvnbItR0AUrxqfvnbJR0BDyfvnbItEh3ENR0BKUcrAgxJvR0AXrhR64UeuR0BDTZFocrAhh3EOR0BJ4m6XjU/fR0AGuFHrhR64R0BD5gQYk3S8h3EPR0BJI/fO2RaHR0AR2yLQ5WBCR0BD3bItDlYEh3EQR0BKVqfvnbItR0AUrxqfvnbJR0BDyfvnbItEh3ERR0BJ5DlYEGJOR0ASa4UeuFHsR0BDteNT987Zh3ESR0BJ5DlYEGJOR0ASa4UeuFHsR0BDteNT987Zh3ETZVUGYWN0aXZlcRRLAHVLAX1xFShLAF1xFihHQEppN0vGp/BHQA1BiTdLxqhHQEIrxqfvnbKHcRdHQEpszMzMzM1HQBacrAgxJulHQEKw5WBBiTeHcRhHQEpwQYk3S8dHQB6XjU/fO2RHQEM141P3ztmHcRlHQElSLQ5WBBlHQB8p++dsi0RHQEMaXjU/fO6HcRpHQEg0GJN0vGpHQB+8an752yNHQEL+uFHrhR+HcRtHQEgwgxJul41HQBfBiTdLxqhHQEJ5ul41P32HcRxHQEgs7ZFocrBHQA+LQ5WBBiVHQEH0m6XjU/iHcR1HQElLItDlYEJHQA5mZmZmZmZHQEIQQYk3S8eHcR5HQElOl41P3ztHQBcvGp++dslHQEKVP3ztkWiHcR9laBRLAHV1Lg=='))
	surfInfo = {'category': (0, None, {}), 'probeRadius': (0, None, {}), 'pointSize': (0, None, {}), 'name': [], 'density': (0, None, {}), 'colorMode': (0, None, {}), 'useLighting': (0, None, {}), 'transparencyBlendMode': (0, None, {}), 'molecule': [], 'smoothLines': (0, None, {}), 'lineWidth': (0, None, {}), 'allComponents': (0, None, {}), 'twoSidedLighting': (0, None, {}), 'customVisibility': [], 'drawMode': (0, None, {}), 'display': (0, None, {}), 'customColors': []}
	vrmlInfo = {'subid': (3, 0, {}), 'display': (3, False, {True: [0]}), 'id': (3, 1, {2: [1], 3: [2]}), 'vrmlString': ['.comment a1 = d-hat; reddish purple (Okabe-Ito palette)\n.color 0.800 0.475 0.655\n.transparency 0.0\n.arrow   50.357    4.766   39.150   50.614    5.796   37.166 0.045 0.140 0.84\n.arrow   52.639    5.920   38.606   52.569    7.293   36.825 0.045 0.140 0.84\n\n.comment a2; bluish green\n.color 0.000 0.620 0.451\n.transparency 0.0\n.arrow   50.357    4.766   39.150   48.122    4.909   38.934 0.045 0.140 0.84\n.arrow   52.639    5.920   38.606   54.564    6.878   39.269 0.045 0.140 0.84\n\n.comment a3 = a1 cross a2; vermillion\n.color 0.835 0.369 0.000\n.transparency 0.0\n.arrow   50.357    4.766   39.150   50.385    6.762   40.189 0.045 0.140 0.84\n.arrow   52.639    5.920   38.606   53.802    4.417   37.402 0.045 0.140 0.84\n\n.comment c-hat from residue centroid to backbone N; auxiliary vector, visually subordinate\n.color 0.50 0.50 0.50\n.transparency 0.0\n.arrow   51.783    4.605   39.421   50.357    4.766   39.150 0.025 0.20 0.7\n.arrow   51.783    4.605   39.421   52.639    5.920   38.606 0.025 0.20 0.7\n\n.comment Thin guide from polar-base position (mean of bonded heavy atoms) to N\n.color 0.500 0.500 0.500\n.transparency 0.0\n.cylinder 50.281 4.464 39.732 50.357 4.766 39.150 0.018',
'#VRML V2.0 utf8\nTransform {\n\ttranslation 52.849500 5.652500 37.381500\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 -0.013780\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 0.480206\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.700000 0.700000 0.700000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cylinder {\n\t\t\t\t\t\t\t\tradius 0.025000\n\t\t\t\t\t\t\t\theight 4.500372\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 50.642000 7.791000 38.205500\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 1.506901\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 -0.095928\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.700000 0.700000 0.700000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cylinder {\n\t\t\t\t\t\t\t\tradius 0.025000\n\t\t\t\t\t\t\t\theight 4.499829\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 48.379000 5.938500 36.950500\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 3.127562\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 -0.480205\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.700000 0.700000 0.700000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cylinder {\n\t\t\t\t\t\t\t\tradius 0.025000\n\t\t\t\t\t\t\t\theight 4.500384\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 50.586500 3.800000 36.126500\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 -1.634677\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 0.095907\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.700000 0.700000 0.700000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cylinder {\n\t\t\t\t\t\t\t\tradius 0.025000\n\t\t\t\t\t\t\t\theight 4.500822\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 52.822000 3.657000 36.342000\n\tchildren [\n\n\n\t\tShape {\n\t\t\tappearance Appearance {\n\t\t\t\tmaterial Material {\n\t\t\t\t\tambientIntensity 1\n\t\t\t\t\tdiffuseColor 0.400000 0.400000 0.400000\n\t\t\t\t\ttransparency 0.000000\n\t\t\t\t}\n\t\t\t}\n\t\t\tgeometry Sphere {\n\t\t\t\tradius 0.200000\n\t\t\t}\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 52.877000 7.648000 38.421000\n\tchildren [\n\n\n\t\tShape {\n\t\t\tappearance Appearance {\n\t\t\t\tmaterial Material {\n\t\t\t\t\tambientIntensity 1\n\t\t\t\t\tdiffuseColor 0.400000 0.400000 0.400000\n\t\t\t\t\ttransparency 0.000000\n\t\t\t\t}\n\t\t\t}\n\t\t\tgeometry Sphere {\n\t\t\t\tradius 0.200000\n\t\t\t}\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 48.351000 3.943000 35.911000\n\tchildren [\n\n\n\t\tShape {\n\t\t\tappearance Appearance {\n\t\t\t\tmaterial Material {\n\t\t\t\t\tambientIntensity 1\n\t\t\t\t\tdiffuseColor 0.400000 0.400000 0.400000\n\t\t\t\t\ttransparency 0.000000\n\t\t\t\t}\n\t\t\t}\n\t\t\tgeometry Sphere {\n\t\t\t\tradius 0.200000\n\t\t\t}\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 48.407000 7.934000 37.990000\n\tchildren [\n\n\n\t\tShape {\n\t\t\tappearance Appearance {\n\t\t\t\tmaterial Material {\n\t\t\t\t\tambientIntensity 1\n\t\t\t\t\tdiffuseColor 0.400000 0.400000 0.400000\n\t\t\t\t\ttransparency 0.000000\n\t\t\t\t}\n\t\t\t}\n\t\t\tgeometry Sphere {\n\t\t\t\tradius 0.200000\n\t\t\t}\n\t\t}\n\t]\n}',
'#VRML V2.0 utf8\nTransform {\n\ttranslation 50.464940 5.198600 38.316720\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 -0.244522\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 -1.079488\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.800000 0.475000 0.655000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cylinder {\n\t\t\t\t\t\t\t\tradius 0.045000\n\t\t\t\t\t\t\t\theight 1.890132\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 50.593440 5.713600 37.324720\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 -0.244522\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 -1.079488\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.800000 0.475000 0.655000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cone {\n\t\t\t\t\t\t\t\tbottomRadius 0.140000\n\t\t\t\t\t\t\t\theight 0.360025\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 49.418300 4.826060 39.059280\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 1.506901\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 -0.096150\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.000000 0.620000 0.451000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cylinder {\n\t\t\t\t\t\t\t\tradius 0.045000\n\t\t\t\t\t\t\t\theight 1.889968\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 48.300800 4.897560 38.951280\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 1.506901\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 -0.096150\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.000000 0.620000 0.451000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cone {\n\t\t\t\t\t\t\t\tbottomRadius 0.140000\n\t\t\t\t\t\t\t\theight 0.359994\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 50.368760 5.604320 39.586380\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 -0.014027\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 0.479905\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.835000 0.369000 0.000000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cylinder {\n\t\t\t\t\t\t\t\tradius 0.045000\n\t\t\t\t\t\t\t\theight 1.890340\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}\nTransform {\n\ttranslation 50.382760 6.602320 40.105880\n\tchildren [\n\t\tTransform {\n\t\t\trotation 0 0 1 -0.014027\n\t\t\tchildren [\n\t\t\t\tTransform {\n\t\t\t\t\trotation 1 0 0 0.479905\n\t\t\t\t\tchildren [\n\n\n\t\t\t\t\t\tShape {\n\t\t\t\t\t\t\tappearance Appearance {\n\t\t\t\t\t\t\t\tmaterial Material {\n\t\t\t\t\t\t\t\t\tambientIntensity 1\n\t\t\t\t\t\t\t\t\tdiffuseColor 0.835000 0.369000 0.000000\n\t\t\t\t\t\t\t\t\ttransparency 0.000000\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\tgeometry Cone {\n\t\t\t\t\t\t\t\tbottomRadius 0.140000\n\t\t\t\t\t\t\t\theight 0.360065\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t]\n\t\t\t\t}\n\t\t\t]\n\t\t}\n\t]\n}'],
'name': (3, u'C:\\Users\\user\\Desktop\\WatGNN_all\\WatGNN-main\\watgnn\\A_axis.bild', {u'C:\\Users\\user\\Desktop\\WatGNN_all\\WatGNN-main\\watgnn\\x_probe.bild': [1], u'C:\\Users\\user\\Desktop\\WatGNN_all\\WatGNN-main\\watgnn\\B_axis.bild': [2]})}
	colors = {u'': ((0.780392, 0, 0.4), 1, u''), u'Ru': ((0.141176, 0.560784, 0.560784), 1, u'default'), u'Re': ((0.14902, 0.490196, 0.670588), 1, u'default'), u'Rf': ((0.8, 0, 0.34902), 1, u'default'), u'Ra': ((0, 0.490196, 0), 1, u'default'), u'Rb': ((0.439216, 0.180392, 0.690196), 1, u'default'), u'Rn': ((0.258824, 0.509804, 0.588235), 1, u'default'), u'Rh': ((0.0392157, 0.490196, 0.54902), 1, u'default'), u'Be': ((0.760784, 1, 0), 1, u'default'), u'Ba': ((0, 0.788235, 0), 1, u'default'), u'Bh': ((0.878431, 0, 0.219608), 1, u'default'), u'Bi': ((0.619608, 0.309804, 0.709804), 1, u'default'), u'Bk': ((0.541176, 0.309804, 0.890196), 1, u'default'), u'Br': ((0.65098, 0.160784, 0.160784), 1, u'default'), u'orange': ((1, 0.498039, 0), 1, u'default'), u'H': ((1, 1, 1), 1, u'default'), u'K': ((0.560784, 0.25098, 0.831373), 1, u'default'), u'P': ((1, 0.501961, 0), 1, u'default'), u'Os': ((0.14902, 0.4, 0.588235), 1, u'default'), u'Ge': ((0.4, 0.560784, 0.560784), 1, u'default'), u'Gd': ((0.270588, 1, 0.780392), 1, u'default'), u'Ga': ((0.760784, 0.560784, 0.560784), 1, u'default'),
u'Pr': ((0.85098, 1, 0.780392), 1, u'default'), u'Pt': ((0.815686, 0.815686, 0.878431), 1, u'default'), u'Pu': ((0, 0.419608, 1), 1, u'default'), u'Mg': ((0.541176, 1, 0), 1, u'default'), u'Pb': ((0.341176, 0.34902, 0.380392), 1, u'default'), u'Pa': ((0, 0.631373, 1), 1, u'default'), u'Pd': ((0, 0.411765, 0.521569), 1, u'default'), u'Cd': ((1, 0.85098, 0.560784), 1, u'default'), u'Po': ((0.670588, 0.360784, 0), 1, u'default'), u'Pm': ((0.639216, 1, 0.780392), 1, u'default'), u'Hs': ((0.901961, 0, 0.180392), 1, u'default'), u'Ho': ((0, 1, 0.611765), 1, u'default'), u'Hf': ((0.301961, 0.760784, 1), 1, u'default'), u'Hg': ((0.721569, 0.721569, 0.815686), 1, u'default'), u'He': ((0.85098, 1, 1), 1, u'default'), u'Md': ((0.701961, 0.0509804, 0.65098), 1, u'default'), u'C': ((0.564706, 0.564706, 0.564706), 1, u'default'), u'dim gray': ((0.411765, 0.411765, 0.411765), 1, u'default'), u'Mo': ((0.329412, 0.709804, 0.709804), 1, u'default'), u'Mn': ((0.611765, 0.478431, 0.780392), 1, u'default'), u'O': ((1, 0.0509804, 0.0509804), 1, u'default'), u'Zr': ((0.580392, 0.878431, 0.878431), 1, u'default'),
u'S': ((1, 1, 0.188235), 1, u'default'), u'W': ((0.129412, 0.580392, 0.839216), 1, u'default'), u'Zn': ((0.490196, 0.501961, 0.690196), 1, u'default'), u'Mt': ((0.921569, 0, 0.14902), 1, u'default'), u'plum': ((0.866667, 0.627451, 0.866667), 1, u'default'), u'Eu': ((0.380392, 1, 0.780392), 1, u'default'), u'Es': ((0.701961, 0.121569, 0.831373), 1, u'default'), u'Er': ((0, 0.901961, 0.458824), 1, u'default'), u'Ni': ((0.313725, 0.815686, 0.313725), 1, u'default'), u'No': ((0.741176, 0.0509804, 0.529412), 1, u'default'), u'Na': ((0.670588, 0.360784, 0.94902), 1, u'default'), u'Nb': ((0.45098, 0.760784, 0.788235), 1, u'default'), u'Nd': ((0.780392, 1, 0.780392), 1, u'default'), u'Ne': ((0.701961, 0.890196, 0.960784), 1, u'default'), u'Np': ((0, 0.501961, 1), 1, u'default'), u'Fr': ((0.258824, 0, 0.4), 1, u'default'), u'Fe': ((0.878431, 0.4, 0.2), 1, u'default'), u'Fm': ((0.701961, 0.121569, 0.729412), 1, u'default'), u'B': ((1, 0.709804, 0.709804), 1, u'default'), u'F': ((0.564706, 0.878431, 0.313725), 1, u'default'), u'Sr': ((0, 1, 0), 1, u'default'), u'N': ((0.188235, 0.313725, 0.972549), 1, u'default'),
u'Kr': ((0.360784, 0.721569, 0.819608), 1, u'default'), u'Si': ((0.941176, 0.784314, 0.627451), 1, u'default'), u'Sn': ((0.4, 0.501961, 0.501961), 1, u'default'), u'Sm': ((0.560784, 1, 0.780392), 1, u'default'), u'V': ((0.65098, 0.65098, 0.670588), 1, u'default'), u'Sc': ((0.901961, 0.901961, 0.901961), 1, u'default'), u'Sb': ((0.619608, 0.388235, 0.709804), 1, u'default'), u'Sg': ((0.85098, 0, 0.270588), 1, u'default'), u'Se': ((1, 0.631373, 0), 1, u'default'), u'Co': ((0.941176, 0.564706, 0.627451), 1, u'default'), u'Cm': ((0.470588, 0.360784, 0.890196), 1, u'default'), u'Cl': ((0.121569, 0.941176, 0.121569), 1, u'default'), u'Ca': ((0.239216, 1, 0), 1, u'default'), u'Cf': ((0.631373, 0.211765, 0.831373), 1, u'default'), u'Ce': ((1, 1, 0.780392), 1, u'default'), u'Xe': ((0.258824, 0.619608, 0.690196), 1, u'default'), u'Tm': ((0, 0.831373, 0.321569), 1, u'default'), u'Cs': ((0.341176, 0.0901961, 0.560784), 1, u'default'), u'Cr': ((0.541176, 0.6, 0.780392), 1, u'default'), u'Cu': ((0.784314, 0.501961, 0.2), 1, u'default'), u'La': ((0.439216, 0.831373, 1), 1, u'default'),
u'Li': ((0.8, 0.501961, 1), 1, u'default'), u'Tl': ((0.65098, 0.329412, 0.301961), 1, u'default'), u'Lu': ((0, 0.670588, 0.141176), 1, u'default'), u'Lr': ((0.780392, 0, 0.4), 1, u'default'), u'Th': ((0, 0.729412, 1), 1, u'default'), u'Ti': ((0.74902, 0.760784, 0.780392), 1, u'default'), u'tan': ((0.823529, 0.705882, 0.54902), 1, u'default'), u'Te': ((0.831373, 0.478431, 0), 1, u'default'), u'Tb': ((0.188235, 1, 0.780392), 1, u'default'), u'Tc': ((0.231373, 0.619608, 0.619608), 1, u'default'), u'Ta': ((0.301961, 0.65098, 1), 1, u'default'), u'Yb': ((0, 0.74902, 0.219608), 1, u'default'), u'Db': ((0.819608, 0, 0.309804), 1, u'default'), u'Dy': ((0.121569, 1, 0.780392), 1, u'default'), u'I': ((0.580392, 0, 0.580392), 1, u'default'), u'salmon': ((0.980392, 0.501961, 0.447059), 1, u'default'), u'medium purple': ((0.576471, 0.439216, 0.858824), 1, u'default'), u'U': ((0, 0.560784, 1), 1, u'default'), u'Y': ((0.580392, 1, 1), 1, u'default'), u'Ac': ((0.439216, 0.670588, 0.980392), 1, u'default'), u'Ag': ((0.752941, 0.752941, 0.752941), 1, u'default'), u'Ir': ((0.0901961, 0.329412, 0.529412), 1, u'default'),
u'Am': ((0.329412, 0.360784, 0.94902), 1, u'default'), u'Al': ((0.74902, 0.65098, 0.65098), 1, u'default'), u'As': ((0.741176, 0.501961, 0.890196), 1, u'default'), u'Ar': ((0.501961, 0.819608, 0.890196), 1, u'default'), u'Au': ((1, 0.819608, 0.137255), 1, u'default'), u'At': ((0.458824, 0.309804, 0.270588), 1, u'default'), u'In': ((0.65098, 0.458824, 0.45098), 1, u'default'), u'light gray': ((0.827451, 0.827451, 0.827451), 1, u'default')}
	materials = {u'': ((0.85, 0.85, 0.85), 30), u'default': ((0.85, 0.85, 0.85), 30)}
	pbInfo = {'category': [u'distance monitor'], 'bondInfo': [{'color': (1, None, {}), 'atoms': [[37, 31]], 'label': (1, u'4.50\xc5', {}), 'halfbond': (1, False, {}), 'labelColor': (1, None, {}), 'labelOffset': (1, chimera.Vector(-1e+99, 0.0, 0.0), {}), 'drawMode': (1, 0, {}), 'display': (1, 2, {})}], 'lineType': (1, 1, {}), 'color': (1, 13, {}), 'optional': {'fixedLabels': (True, False, (1, 0, {}))}, 'display': (1, True, {}), 'showStubBonds': (1, False, {}), 'lineWidth': (1, 1, {}), 'stickScale': (1, 1, {}), 'id': [-2]}
	modelAssociations = {}
	colorInfo = (16, (u'', (0, 0, 0, 0)), {(u'H', (1, 1, 1, 1)): [12], (u'green', (0, 1, 0, 1)): [15], (u'N', (0.188235, 0.313725, 0.972549, 1)): [10], (u'', (1, 0, 0, 0.333)): [9], (u'magenta', (1, 0, 1, 1)): [2], (u'', (1, 1, 1, 1)): [14], (u'dim gray', (0.411765, 0.411765, 0.411765, 1)): [3], (u'O', (1, 0.0509804, 0.0509804, 1)): [11], (u'', (0.411765, 0.411765, 0.411765, 0.333)): [8], (u'tan', (0.823529, 0.705882, 0.54902, 1)): [0], (u'', (0, 0, 0, 1)): [13], (u'', (0.52381, 1, 1, 0)): [6], (u'salmon', (0.980392, 0.501961, 0.447059, 1)): [1], (u'', (0, 0, 1, 0.333)): [7]})
	viewerInfo = {'cameraAttrs': {'center': (49.828000002384, 5.213, 39.323), 'fieldOfView': 26.507800948255, 'nearFar': (43.249982556879, -27.156212122763), 'ortho': True, 'eyeSeparation': 50.8, 'focal': 39.323}, 'viewerAttrs': {'silhouetteColor': None, 'clipping': True, 'showSilhouette': False, 'showShadows': False, 'viewSize': 8.0728944987978, 'labelsOnTop': True, 'depthCueRange': (0.5, 1), 'silhouetteWidth': 2, 'singleLayerTransparency': True, 'shadowTextureSize': 2048, 'backgroundImage': [None, 1, 2, 1, 0, 0], 'backgroundGradient': [('Chimera default', [(1, 1, 1, 1), (0, 0, 1, 1)], 1), 1, 0, 0], 'depthCue': True, 'highlight': 0, 'scaleFactor': 2.4488153955446, 'angleDependentTransparency': True, 'backgroundMethod': 0}, 'viewerHL': 15, 'cameraMode': 'mono', 'detail': 5, 'viewerFog': None, 'viewerBG': 14}

	replyobj.status("Initializing session restore...", blankAfter=0,
		secondary=True)
	from SimpleSession.versions.v65 import expandSummary
	init(dict(enumerate(expandSummary(colorInfo))))
	replyobj.status("Restoring colors...", blankAfter=0,
		secondary=True)
	restoreColors(colors, materials)
	replyobj.status("Restoring molecules...", blankAfter=0,
		secondary=True)
	restoreMolecules(molInfo, resInfo, atomInfo, bondInfo, crdInfo)
	replyobj.status("Restoring surfaces...", blankAfter=0,
		secondary=True)
	restoreSurfaces(surfInfo)
	replyobj.status("Restoring VRML models...", blankAfter=0,
		secondary=True)
	restoreVRML(vrmlInfo)
	replyobj.status("Restoring pseudobond groups...", blankAfter=0,
		secondary=True)
	restorePseudoBondGroups(pbInfo)
	replyobj.status("Restoring model associations...", blankAfter=0,
		secondary=True)
	restoreModelAssociations(modelAssociations)
	replyobj.status("Restoring camera...", blankAfter=0,
		secondary=True)
	restoreViewer(viewerInfo)

try:
	restoreCoreModels()
except:
	reportRestoreError("Error restoring core models")

	replyobj.status("Restoring extension info...", blankAfter=0,
		secondary=True)


try:
	import StructMeasure
	from StructMeasure.DistMonitor import restoreDistances
	registerAfterModelsCB(restoreDistances, 1)
except:
	reportRestoreError("Error restoring distances in session")


def restoreMidasBase():
	formattedPositions = {'session-start': (2.4488153955446, 8.0728944987978, (49.828000002384, 5.213, 39.323), (43.249982556879, -27.156212122763), 39.323, {(3, 0): ((100.01244762424, 21.486541462109, 73.444219091369), (0.09249351563784361, 0.914629159359708, -0.39357140446671923, 177.27303775512982)), (2, 0): ((100.01244762424, 21.486541462109, 73.444219091369), (0.09249351563784361, 0.914629159359708, -0.39357140446671923, 177.27303775512982)), (1, 0): ((100.01244762424, 21.486541462109, 73.444219091369), (0.09249351563784361, 0.914629159359708, -0.39357140446671923, 177.27303775512982)), (0, 0): ((100.01244762424, 21.486541462109, 73.444219091369), (0.09249351563784361, 0.914629159359708, -0.39357140446671923, 177.27303775512982)), (4, 0): ((100.01244762424, 21.486541462109, 73.444219091369), (0.09249351563784361, 0.914629159359708, -0.39357140446671923, 177.27303775512982))}, {(4, 0, 'Molecule'): (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 5.0), (3, 0, 'VRMLModel'): (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 5.0), (2, 0, 'VRMLModel'): (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 5.0), (0, 0, 'Molecule'): (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 5.0)}, 4, (49.65777340568273, 3.527452261999887, 37.316498645572636), True, 26.507800948255)}
	import Midas
	Midas.restoreMidasBase(formattedPositions)
try:
	restoreMidasBase()
except:
	reportRestoreError('Error restoring Midas base state')


def restoreMidasText():
	from Midas import midas_text
	midas_text.aliases = {}
	midas_text.userSurfCategories = {}

try:
	restoreMidasText()
except:
	reportRestoreError('Error restoring Midas text state')


def restore_cap_attributes():
 cap_attributes = \
  {
   'cap_attributes': [ ],
   'cap_color': None,
   'cap_offset': 0.01,
   'class': 'Caps_State',
   'default_cap_offset': 0.01,
   'mesh_style': False,
   'shown': True,
   'subdivision_factor': 1.0,
   'version': 1,
  }
 import SurfaceCap.session
 SurfaceCap.session.restore_cap_attributes(cap_attributes)
registerAfterModelsCB(restore_cap_attributes)


def restore_volume_data():
 volume_data_state = \
  {
   'class': 'Volume_Manager_State',
   'data_and_regions_state': [ ],
   'version': 2,
  }
 from VolumeViewer import session
 session.restore_volume_data_state(volume_data_state)

try:
  restore_volume_data()
except:
  reportRestoreError('Error restoring volume data')

geomData = {'AxisManager': {}, 'CentroidManager': {}, 'PlaneManager': {}}

try:
	from StructMeasure.Geometry import geomManager
	geomManager._restoreSession(geomData)
except:
	reportRestoreError("Error restoring geometry objects in session")


def restoreSession_RibbonStyleEditor():
	import SimpleSession
	import RibbonStyleEditor
	userScalings = []
	userXSections = []
	userResidueClasses = []
	residueData = [(2, 'Chimera default', 'rounded', u'amino acid'), (3, 'Chimera default', 'rounded', u'amino acid'), (4, 'Chimera default', 'rounded', u'unknown'), (5, 'Chimera default', 'rounded', u'unknown'), (6, 'Chimera default', 'rounded', u'unknown'), (7, 'Chimera default', 'rounded', u'unknown'), (8, 'Chimera default', 'rounded', u'unknown'), (9, 'Chimera default', 'rounded', u'unknown'), (10, 'Chimera default', 'rounded', u'unknown'), (11, 'Chimera default', 'rounded', u'unknown'), (12, 'Chimera default', 'rounded', u'unknown'), (13, 'Chimera default', 'rounded', u'unknown'), (14, 'Chimera default', 'rounded', u'unknown')]
	flags = RibbonStyleEditor.NucleicDefault1
	SimpleSession.registerAfterModelsCB(RibbonStyleEditor.restoreState,
				(userScalings, userXSections,
				userResidueClasses, residueData, flags))
try:
	restoreSession_RibbonStyleEditor()
except:
	reportRestoreError("Error restoring RibbonStyleEditor state")

trPickle = 'gAJjQW5pbWF0ZS5UcmFuc2l0aW9ucwpUcmFuc2l0aW9ucwpxASmBcQJ9cQMoVQxjdXN0b21fc2NlbmVxBGNBbmltYXRlLlRyYW5zaXRpb24KVHJhbnNpdGlvbgpxBSmBcQZ9cQcoVQZmcmFtZXNxCEsBVQ1kaXNjcmV0ZUZyYW1lcQlLAVUKcHJvcGVydGllc3EKXXELVQNhbGxxDGFVBG5hbWVxDVUMY3VzdG9tX3NjZW5lcQ5VBG1vZGVxD1UGbGluZWFycRB1YlUIa2V5ZnJhbWVxEWgFKYFxEn1xEyhoCEsUaAlLAWgKXXEUaAxhaA1VCGtleWZyYW1lcRVoD2gQdWJVBXNjZW5lcRZoBSmBcRd9cRgoaAhLAWgJSwFoCl1xGWgMYWgNVQVzY2VuZXEaaA9oEHVidWIu'
scPickle = 'gAJjQW5pbWF0ZS5TY2VuZXMKU2NlbmVzCnEBKYFxAn1xA1UHbWFwX2lkc3EEfXNiLg=='
kfPickle = 'gAJjQW5pbWF0ZS5LZXlmcmFtZXMKS2V5ZnJhbWVzCnEBKYFxAn1xA1UHZW50cmllc3EEXXEFc2Iu'
def restoreAnimation():
	'A method to unpickle and restore animation objects'
	# Scenes must be unpickled after restoring transitions, because each
	# scene links to a 'scene' transition. Likewise, keyframes must be 
	# unpickled after restoring scenes, because each keyframe links to a scene.
	# The unpickle process is left to the restore* functions, it's 
	# important that it doesn't happen prior to calling those functions.
	import SimpleSession
	from Animate.Session import restoreTransitions
	from Animate.Session import restoreScenes
	from Animate.Session import restoreKeyframes
	SimpleSession.registerAfterModelsCB(restoreTransitions, trPickle)
	SimpleSession.registerAfterModelsCB(restoreScenes, scPickle)
	SimpleSession.registerAfterModelsCB(restoreKeyframes, kfPickle)
try:
	restoreAnimation()
except:
	reportRestoreError('Error in Animate.Session')

def restoreLightController():
	import Lighting
	Lighting._setFromParams({'ratio': 1.25, 'brightness': 1.16, 'material': [30.0, (0.85, 0.85, 0.85), 1.0], 'back': [(0.3574067443365933, 0.6604015517481455, -0.6604015517481456), (1.0, 1.0, 1.0), 0.0], 'mode': 'two-point', 'key': [(-0.3574067443365933, 0.6604015517481455, 0.6604015517481456), (1.0, 1.0, 1.0), 1.0], 'contrast': 0.83, 'fill': [(0.2505628070857316, 0.2505628070857316, 0.9351131265310294), (1.0, 1.0, 1.0), 0.0]})
try:
	restoreLightController()
except:
	reportRestoreError("Error restoring lighting parameters")


def restoreRemainder():
	from SimpleSession.versions.v65 import restoreWindowSize, \
	     restoreOpenStates, restoreSelections, restoreFontInfo, \
	     restoreOpenModelsAttrs, restoreModelClip, restoreSilhouettes

	curSelIds =  []
	savedSels = []
	openModelsAttrs = { 'cofrMethod': 4 }
	windowSize = (720, 960)
	xformMap = {0: (((0.092493515637844, 0.91462915935971, -0.39357140446672), 177.27303775513), (100.01244762424, 21.486541462109, 73.444219091369), True), 1: (((0.092493515637844, 0.91462915935971, -0.39357140446672), 177.27303775513), (100.01244762424, 21.486541462109, 73.444219091369), True), 51: (((0.092493515637844, 0.91462915935971, -0.39357140446672), 177.27303775513), (100.01244762424, 21.486541462109, 73.444219091369), True), 52: (((0.092493515637844, 0.91462915935971, -0.39357140446672), 177.27303775513), (100.01244762424, 21.486541462109, 73.444219091369), True), 53: (((0.092493515637844, 0.91462915935971, -0.39357140446672), 177.27303775513), (100.01244762424, 21.486541462109, 73.444219091369), True)}
	fontInfo = {'face': ('Serif', 'Bold', 36)}
	clipPlaneInfo = {}
	silhouettes = {0: True, 1: True, 51: True, 52: True, 53: True, 55: True}

	replyobj.status("Restoring window...", blankAfter=0,
		secondary=True)
	restoreWindowSize(windowSize)
	replyobj.status("Restoring open states...", blankAfter=0,
		secondary=True)
	restoreOpenStates(xformMap)
	replyobj.status("Restoring font info...", blankAfter=0,
		secondary=True)
	restoreFontInfo(fontInfo)
	replyobj.status("Restoring selections...", blankAfter=0,
		secondary=True)
	restoreSelections(curSelIds, savedSels)
	replyobj.status("Restoring openModel attributes...", blankAfter=0,
		secondary=True)
	restoreOpenModelsAttrs(openModelsAttrs)
	replyobj.status("Restoring model clipping...", blankAfter=0,
		secondary=True)
	restoreModelClip(clipPlaneInfo)
	replyobj.status("Restoring per-model silhouettes...", blankAfter=0,
		secondary=True)
	restoreSilhouettes(silhouettes)

	replyobj.status("Restoring remaining extension info...", blankAfter=0,
		secondary=True)
try:
	restoreRemainder()
except:
	reportRestoreError("Error restoring post-model state")
from SimpleSession.versions.v65 import makeAfterModelsCBs
makeAfterModelsCBs()

from SimpleSession.versions.v65 import endRestore
replyobj.status('Finishing restore...', blankAfter=0, secondary=True)
endRestore({})
replyobj.status('', secondary=True)
replyobj.status('Restore finished.')

