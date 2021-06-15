#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import struct
import sys
import os
import datetime
import argparse
import textwrap

_colorspacesignatures = {
    "XYZ ": "nCIEXYZ or PCSXYZ",
    "Lab ": "CIELAB or PCSLAB",
    "Luv ": "CIELUV",
    "YCbr": "YCbCr",
    "Yxy ": "CIEYxy",
    "RGB ": "RGB",
    "GRAY": "Gray",
    "HSV ": "HSV",
    "HLS ": "HLS",
    "CMYK": "CMYK",
    "CMY ": "CMY",
    "2CLR": "2 colour",
    "3CLR": "3 colour",
    "4CLR": "4 colour",
    "5CLR": "5 colour",
    "6CLR": "6 colour",
    "7CLR": "7 colour",
    "8CLR": "8 colour",
    "9CLR": "9 colour",
    "ACLR": "10 colour",
    "BCLR": "11 colour",
    "CCLR": "12 colour",
    "DCLR": "13 colour",
    "ECLR": "14 colour",
    "FCLR": "15 colour"
}

_profileclasssignatures = {
    "scnr": "Input device profile",
    "mntr": "Display device profile",
    "prtr": "Output device profile",
    "link": "DeviceLink profile",
    "spac": "ColorSpace profile",
    "abst": "Abstract profile",
    "nmcl": "NamedColor profile"
}

_primaryplatformsignatures = {
    "APPL": "Apple Computer, Inc.",
    "MSFT": "Microsoft Corporation",
    "SGI ": "Silicon Graphics, Inc.",
    "SUNW": "Sun Microsystems, Inc."
}

_renderingintenttable = {
    0: "Perceptual",
    1: "Media-relative colorimetric",
    2: "Saturation",
    3: "ICC-absolute colorimetric"
}

_parametriccurvetypetable = {
    0: {
            "function": "Y = X**g",
            "fieldlength": 4,
            "parameters": {
                "g": None
            }
    },
    1: {
            "function": "Y = (aX + b)**g if X >= -b/a else Y = 0",
            "fieldlength": 12,
            "parameters": {
                "g": None,
                "a": None,
                "b": None
            }
    },
    2: {
            "function": "Y = (aX + b)**g + c if X >= -b/a else Y = c",
            "fieldlength": 16,
            "parameters": {
                "g": None,
                "a": None,
                "b": None,
                "c": None
            }
    },
    3: {
            "function": "Y = (aX + b)**g if X >= d else Y = cX",
            "fieldlength": 20,
            "parameters": {
                "g": None,
                "a": None,
                "b": None,
                "c": None,
                "d": None
            }
    },
    4: {
            "function": "Y = (aX + b)**g if X >= d else Y = cX + f",
            "fieldlength": 28,
            "parameters": {
                "g": None,
                "a": None,
                "b": None,
                "c": None,
                "d": None,
                "e": None,
                "f": None
            }
    }
}


class ICCFileError(ValueError):
    def __init__(self, message):
        self.message = message


def as_numeric(obj, as_type=numpy.float64):
    try:
        return as_type(obj)
    except TypeError:
        return obj


def fs15f16(x):
    """Convert float to ICC s15Fixed16Number (as a Python ``int``)."""
    return int(round(x * 2**16))


def unpack_s15Fixed16Number(s):
    """Convert buffer of ICC s15Fixed16 to array of float."""
    return as_numeric(numpy.divide(
        numpy.frombuffer(s, numpy.dtype(">i4"), len(s) // 4), 2**16))


def unpack_string(s, codec="utf-8"):
    """Convert sequence of n bytes into a string"""
    return struct.unpack("{}s".format(len(s)), s)[0].decode(codec)


def unpack_tagSignature(s):
    """Convert sequence of 4 bytes into a string."""
    return struct.unpack("4s", s)[0].decode("utf-8")


def unpack_uInt8Number(s):
    """Convert sequence of 1 byte into an 8 bit unsigned int."""
    return struct.unpack(">B", s)[0]


def unpack_uInt16Number(s):
    """Convert sequence of 2 bytes into a big endian 16 bit unsigned int."""
    return struct.unpack(">H", s)[0]


def unpack_uInt32Number(s):
    """Convert sequence of 4 bytes into a big endian 32 bit unsigned int."""
    return struct.unpack(">I", s)[0]


def unpack_u8Fixed8Number(s):
    """Convert buffer of ICC u8Fixed8Number to array of float."""
    return as_numeric(numpy.divide(
            numpy.frombuffer(s, numpy.dtype(">u2"), len(s) // 2), 2**8))


class XYZNumber:
    def __init__(self):
        self._XYZ = numpy.array([0., 0., 0.])
        self._xyY = numpy.array([0., 0., 0.])

    @property
    def XYZ(self):
        return self._XYZ

    @property
    def X(self):
        return self._XYZ[0]

    @property
    def Y(self):
        return self._XYZ[1]

    @property
    def Z(self):
        return self._XYZ[2]

    @property
    def xyY(self):
        return self._xyY

    def read(self, buffer):
        try:
            self._XYZ = unpack_s15Fixed16Number(buffer)
            XYZ_sum = numpy.sum(self._XYZ)
            self._xyY[0] = numpy.divide(
                self._XYZ[0], XYZ_sum, where=XYZ_sum != 0)
            self._xyY[1] = numpy.divide(
                self._XYZ[1], XYZ_sum, where=XYZ_sum != 0)
            self._xyY[2] = self._XYZ[1]

        except Exception:
            raise ICCFileError("file doesn't appear to have a profile size")

    def __repr__(self):
        return "<class '{0}(XYZ({1}))'>".format(
            self.__class__.__name__, self._XYZ)

    def __str__(self):
        return "[X: {} Y: {} Z: {}][x: {} y: {} Y: {}]".format(
            "{:<.15f}{}".format(self._XYZ[0], ","),
            "{:<.15f}{}".format(self._XYZ[1], ","),
            "{:<.15f}".format(self._XYZ[2]),
            "{:<.15f}{}".format(self._xyY[0], ","),
            "{:<.15f}{}".format(self._xyY[1], ","),
            "{:<.15f}".format(self._xyY[2])
        )


class iccProfileElement:
    def __init__(self, offset, length):
        self._slice = slice(offset, offset + length)
        self._value = None

    @property
    def slice(self):
        return self._slice

    def __repr__(self):
        return "<class '{0}({1})'>".format(
            self.__class__.__name__, self._slice
        )


# All ICC types are defined below, based off of the iccProfileElement.
# To add an unsupported type, simply define a new class based off
# of iccProfileElement with the four letter tag signature as the
# first four letters of the class, followed by "Type". For example,
# the desc type is defined as descType. The module will automatically
# parse the function name from iccTag, calling the read() function
# of the defined type.
#
# Identification comment should identify the ICC specified type from
# the specification, with the relevant four byte tag


# multiLocalizedUnicodeType, identifier "mluc"
class mlucRecord(object):
    def __init__(self, lang, country, text):
        self._lang = lang
        self._country = country
        self._text = text

    def __str__(self):
        return "[{}, {}, {}]".format(
            self._lang,
            self._country,
            self._text
        )

class mlucType(iccProfileElement):
    def __init__(self, offset, length, buffer):
        super(mlucType, self).__init__(offset, length)
        self._typesignature = None
        self._reserved = None
        self._recordcount = None
        self._recordsize = None
        self._records = None
        self._description = None
        self.read(buffer)

    def read(self, buffer):
        try:
            texttypebuffer = buffer[self._slice]
            self._typesignature = unpack_tagSignature(texttypebuffer[0:4])
            self._reserved = unpack_uInt32Number(texttypebuffer[4:8])
            self._recordcount = unpack_uInt32Number(texttypebuffer[8:12])
            self._recordsize = unpack_uInt32Number(texttypebuffer[12:16])
            self._records = []

            for idx in range(self._recordcount):
                start = 16 + idx * 12

                lang = unpack_string(texttypebuffer[start:start+2])
                country = unpack_string(texttypebuffer[start+2:start+4])
                size = unpack_uInt32Number(texttypebuffer[start+4:start+8])
                offset = unpack_uInt32Number(texttypebuffer[start+8:start+12])
                text = unpack_string(texttypebuffer[offset:offset+size], codec="utf-16be")
                self._records.append(mlucRecord(lang, country, text))

        except Exception:
            raise ICCFileError("problem loading mlucType")

    def __str__(self):
        return "[\"{}\", {}, {}, {}, {}]".format(
            self._typesignature,
            self._reserved,
            self._recordcount,
            self._recordsize,
            ", ".join([str(r) for r in self._records])
        )


# curveType, identifier "curv"
class curvType(iccProfileElement):
    def __init__(self, offset, length, buffer):
        super(curvType, self).__init__(offset, length)
        self._typesignature = None
        self._reserved = None
        self._entriescount = None
        self._curve = None
        self._curvetype = None
        self.read(buffer)

    @property
    def value(self):
        return self._curve

    @property
    def curvetype(self):
        return self._curvetype

    def read(self, buffer):
        try:
            curvtypebuffer = buffer[self._slice]
            self._typesignature = unpack_tagSignature(curvtypebuffer[0:4])
            self._reserved = unpack_uInt32Number(curvtypebuffer[4:8])
            self._entriescount = unpack_uInt32Number(curvtypebuffer[8:12])

            if self._entriescount == 0:
                # Curve type is an identity
                self._curvetype = "Identity Curve"
            elif self._entriescount == 1:
                # Curve type is a pure power function
                self._curvetype = "Power Function"
                self._curve = unpack_u8Fixed8Number(curvtypebuffer[12:14])
            else:
                # Curve is a 1D curve of 16 bit integer entries
                self._curvetype = "1D Curve"

                self._curve = numpy.frombuffer(
                    curvtypebuffer,
                    dtype=">u2",
                    count=self._entriescount,
                    offset=12
                ) / (2**16 - 1)

        except Exception as e:
            raise ICCFileError("problem loading curvType")

    def extract_lut(self, name):
        lut = ""
        lut += "Version 1\n"
        lut += "From 0 1\n"
        lut += "Length {}\n".format(self._curve.shape[0])
        lut += "Components 1\n"
        lut += "{\n"
        lut += "\n".join(["  {:.5f}".format(v) for v in self._curve])
        lut += "\n}"

        with open(name + ".spi1d", "w") as f:
            f.write(lut)

    def __str__(self):
        return "[\"{}\", {}, \"{}\", {}, {}]".format(
            self._typesignature,
            self._reserved,
            self._curvetype,
            self._entriescount,
            self._curve
        )


# parametricCurveType, identifier "para"
class paraType(iccProfileElement):
    def __init__(self, offset, length, buffer):
        super(paraType, self).__init__(offset, length)
        self._typesignature = None
        self._reserved = None
        self._functiontype = None
        self._reservedsecond = None
        self._function = None
        self._parameters = None
        self.read(buffer)

    @property
    def value(self):
        return self._function

    @property
    def parameters(self):
        return self._parameters

    def read(self, buffer):
        try:
            paratypebuffer = buffer[self._slice]
            self._typesignature = unpack_tagSignature(paratypebuffer[0:4])
            self._reserved = unpack_uInt32Number(paratypebuffer[4:8])
            self._functiontype = unpack_uInt16Number(paratypebuffer[8:10])
            self._reservedsecond = unpack_uInt16Number(paratypebuffer[10:12])

            functiontableentry = _parametriccurvetypetable.get(
                self._functiontype, -1)

            self._function = functiontableentry.get(
                "function", "Invalid Function")
            parameters = functiontableentry.get(
                "parameters", "Invalid Parameters")

            self._parameters = {}
            for index, parameter in enumerate(parameters):
                start = (index * 4) + 12
                end = start + 4
                self._parameters[parameter] = \
                    unpack_s15Fixed16Number(paratypebuffer[start:end])

        except Exception as e:
            raise ICCFileError("problem loading paraType")

    def __str__(self):
        return "[\"{}\", {}, {}, {}, \"{}\", {}]".format(
            self._typesignature,
            self._reserved,
            self._functiontype,
            self._reservedsecond,
            self._function,
            self._parameters
        )


# textDescriptionType, identifier "desc"
class descType(iccProfileElement):
    def __init__(self, offset, length, buffer):
        super(descType, self).__init__(offset, length)
        self._typesignature = None
        self._reserved = None
        self._asciicount = None
        self._asciidescription = None
        self._unicodecode = None
        self._unicodecount = None
        self._unicodedescription = None
        self._scriptcodecode = None
        self._scriptcodecount = None
        self._scriptcodedescription = None
        self.read(buffer)

    def read(self, buffer):
        try:
            desctypebuffer = buffer[self._slice]
            self._typesignature = unpack_tagSignature(desctypebuffer[0:4])
            self._reserved = unpack_uInt32Number(desctypebuffer[4:8])
            self._asciicount = unpack_uInt32Number(desctypebuffer[8:12])
            endofascii = 12 + self._asciicount
            self._asciidescription = unpack_string(
                desctypebuffer[12:endofascii]
            )
            self._unicodecode = unpack_uInt32Number(
                desctypebuffer[endofascii:endofascii + 4]
            )
            self._unicodecount = unpack_uInt32Number(
                desctypebuffer[endofascii + 4:endofascii + 8]
            )
            endofunicode = endofascii + self._unicodecount + 8
            self._unicodedescription = unpack_string(
                desctypebuffer[endofascii + 8:endofunicode]
            )
            self._scriptcodecode = unpack_uInt16Number(
                desctypebuffer[endofunicode:endofunicode + 2]
            )
            self._scriptcodecount = unpack_uInt8Number(
                desctypebuffer[endofunicode + 2:endofunicode + 3]
            )
            self._scriptcodedescription = unpack_string(
                desctypebuffer[endofunicode + 3:endofunicode + 70]
            )

        except Exception:
            raise ICCFileError("problem loading descType")

    def __str__(self):
        return "[\"{}\", {}, {}, \"{}\", {}, {}, \"{}\", {}, {}, \"{}\"]" \
            .format(
                self._typesignature,
                self._reserved,
                self._asciicount,
                self._asciidescription,
                self._unicodecode,
                self._unicodecount,
                self._unicodedescription,
                self._scriptcodecode,
                self._scriptcodecount,
                self._scriptcodedescription
            )


# textType, identifier "text"
class textType(iccProfileElement):
    def __init__(self, offset, length, buffer):
        super(textType, self).__init__(offset, length)
        self._typesignature = None
        self._reserved = None
        self._description = None
        self.read(buffer)

    def read(self, buffer):
        try:
            texttypebuffer = buffer[self._slice]
            self._typesignature = unpack_tagSignature(texttypebuffer[0:4])
            self._reserved = unpack_uInt32Number(texttypebuffer[4:8])
            self._description = unpack_string(texttypebuffer[8:])

        except Exception:
            raise ICCFileError("problem loading textType")

    def __str__(self):
        return "[\"{}\", {}, \"{}\"]".format(
            self._typesignature,
            self._reserved,
            textwrap.shorten(self._description, width=256)
        )


class XYZ_Type(iccProfileElement):
    def __init__(self, offset, length, buffer):
        super(XYZ_Type, self).__init__(offset, length)
        self._typesignature = None
        self._reserved = None
        self._XYZ = None
        self.read(buffer)

    @property
    def value(self):
        return self._XYZ

    def read(self, buffer):
        try:
            xyztypebuffer = buffer[self._slice]
            self._typesignature = unpack_tagSignature(xyztypebuffer[0:4])
            self._reserved = unpack_uInt32Number(xyztypebuffer[4:8])

            xyzcount = (self._slice.stop - self._slice.start - 8) // 12
            self._XYZ = numpy.empty(xyzcount, dtype=object)

            for count in range(xyzcount):
                self._XYZ[count] = XYZNumber()

                start = (count * 12) + 8
                stop = ((count + 1) * 12) + 8
                self._XYZ[count].read(xyztypebuffer[start:stop])

        except Exception:
            raise ICCFileError("problem loading XYZ_Type")

    def __str__(self):
        xyzstring = ""
        for index, xyznumber in numpy.ndenumerate(self._XYZ):
            if index[0] == 0:
                xyzstring += str(xyznumber)
            else:
                xyzstring += "\n{:35}{:<}".format(
                    "",
                    str(xyznumber)
                )
        return "[\"{}\", {}, {}]".format(
            self._typesignature,
            self._reserved,
            xyzstring
        )


class sf32Type(iccProfileElement):
    def __init__(self, offset, length, buffer):
        super(sf32Type, self).__init__(offset, length)
        self._typesignature = None
        self._reserved = None
        self._sf32 = None
        self.read(buffer)

    @property
    def value(self):
        return self._sf32

    def read(self, buffer):
        try:
            sf32typebuffer = buffer[self._slice]
            self._typesignature = unpack_tagSignature(sf32typebuffer[0:4])
            self._reserved = unpack_uInt32Number(sf32typebuffer[4:8])

            sf32count = (self._slice.stop - self._slice.start - 8) // 4

            self._sf32 = unpack_s15Fixed16Number(
                sf32typebuffer[8:self._slice.stop]
            )

        except Exception:
            raise ICCFileError("problem loading sf32Type")

    def __str__(self):
        sf32string = ""
        for index, sf32number in numpy.ndenumerate(self._sf32):
            if index[0] == 0:
                sf32string += "[" + str(sf32number)
            else:
                sf32string += ", {:<.15f}".format(sf32number)
        sf32string += "]"
        return "[\"{}\", {}, {}]".format(
            self._typesignature,
            self._reserved,
            sf32string
        )


class iccProfileSize(iccProfileElement):
    def __init__(self):
        super(iccProfileSize, self).__init__(0, 4)
        self._profilesize = None

    @property
    def profilesize(self):
        return self._profilesize

    def read(self, buffer):
        try:
            profilesizebuffer = buffer[self._slice]
            self._profilesize = unpack_uInt32Number(profilesizebuffer)

        except Exception:
            raise ICCFileError("file doesn't appear to have a profile size")

    def __repr__(self):
        return "<class '{0}({1}, profilesize({2}))'>".format(
            self.__class__.__name__, self._slice, self._profilesize
        )

    def __str__(self):
        return "{:>30}{:5}{:<15}".format(
            "Profile Size:",
            "",
            str(self._profilesize)
        )


class iccPreferredCMMType(iccProfileElement):
    def __init__(self):
        super(iccPreferredCMMType, self).__init__(4, 4)
        self._preferredcmmtype = None

    def read(self, buffer):
        try:
            preferredcmmtypebuffer = buffer[self._slice]
            self._preferredcmmtype = unpack_tagSignature(
                preferredcmmtypebuffer)

        except Exception:
            raise ICCFileError("file doesn't appear to have a prefered CMM "
                               "type")

    @property
    def preferredcmmtype(self):
        return self._preferredcmmtype

    def __repr__(self):
        return "<class '{0}({1}, preferredcmmtype('{2}'))'>".format(
            self.__class__.__name__, self._slice, self._preferredcmmtype
        )

    def __str__(self):
        return "{:>30}{:5}\"{:<}\"".format(
            "Preferred CMM Type:",
            "",
            str(self._preferredcmmtype)
        )


class iccProfileVersion(iccProfileElement):
    def __init__(self):
        super(iccProfileVersion, self).__init__(8, 4)
        self._majorVersion = None
        self._minorVersion = None
        self._bugFixVersion = None

    def read(self, buffer):
        try:
            versionbuffer = buffer[self._slice]
            self._majorVersion = struct.unpack("b", versionbuffer[0:1])[0]
            self._minorVersion = struct.unpack("b", versionbuffer[1:2])[0] >> 4
            self._bugFixVersion = struct.unpack("b", versionbuffer[1:2])[0] & \
                0b00001111
            self._reserved = struct.unpack("H", versionbuffer[2:4])[0]

        except Exception:
            raise ICCFileError("file doesn't appear to have a profile version "
                               )

    @property
    def majorVersion(self):
        return self._majorVersion

    @property
    def minorVersion(self):
        return self._minorVersion

    @property
    def bugFixVersion(self):
        return self._bugFixVersion

    def __repr__(self):
        return "<class '{0}({1}, version({2}.{3}.{4}))'>".format(
            self.__class__.__name__, self._slice, self._majorVersion,
            self._minorVersion, self._bugFixVersion
        )

    def __str__(self):
        return "{:>30}{:5}{:<}.{}.{}".format(
            "Profile Version:",
            "",
            str(self._majorVersion),
            str(self._minorVersion),
            str(self._bugFixVersion)
        )


class iccProfileDeviceClass(iccProfileElement):
    def __init__(self):
        super(iccProfileDeviceClass, self).__init__(12, 4)
        self._profiledeviceclass = None
        self._profiledeviceclassdescription = None

    def read(self, buffer):
        try:
            profiledeviceclassbuffer = buffer[self._slice]
            self._profiledeviceclass = unpack_tagSignature(
                profiledeviceclassbuffer[0:4])
            self._profiledeviceclassdescription = \
                _profileclasssignatures.get(self._profiledeviceclass, "None")

        except Exception:
            raise ICCFileError("file doesn't appear to have a profile / "
                               "device class")

    @property
    def profiledeviceclass(self):
        return self._profiledeviceclass

    @property
    def profiledeviceclassdescription(self):
        return self._profiledeviceclassdescription

    def __repr__(self):
        return "<class '{0}({1}, profiledeviceclass('{2}', '{3}'))'>".format(
            self.__class__.__name__, self._slice, self._profiledeviceclass,
            self._profiledeviceclassdescription
        )

    def __str__(self):
        return "{:>30}{:5}\"{:<}\", \"{}\"".format(
            "Profile / Device Class:",
            "",
            str(self._profiledeviceclass),
            str(self._profiledeviceclassdescription)
        )


class iccDataColorSpace(iccProfileElement):
    def __init__(self):
        super(iccDataColorSpace, self).__init__(16, 4)
        self._datacolorspace = None
        self._datacolorspacedescription = None

    def read(self, buffer):
        try:
            datacolorspacebuffer = buffer[self._slice]
            self._datacolorspace = unpack_tagSignature(
                datacolorspacebuffer[0:4])
            self._datacolorspacedescription = \
                _colorspacesignatures.get(self._datacolorspace, "None")

        except Exception:
            raise ICCFileError("file doesn't appear to have a data colorspace")

    @property
    def datacolorspace(self):
        return self._datacolorspace

    @property
    def datacolorspacedescription(self):
        return self._datacolorspacedescription

    def __repr__(self):
        return "<class '{0}({1}, datacolorspace('{2}', '{3}'))'>".format(
            self.__class__.__name__, self._slice, self._datacolorspace,
            self._datacolorspacedescription
        )

    def __str__(self):
        return "{:>30}{:5}\"{:<}\", \"{}\"".format(
            "Profile Data Colorspace:",
            "",
            str(self._datacolorspace),
            str(self._datacolorspacedescription)
        )


class iccPCS(iccProfileElement):
    def __init__(self):
        super(iccPCS, self).__init__(20, 4)
        self._pcs = None
        self._pcsdescription = None

    def read(self, buffer):
        try:
            pcsbuffer = buffer[self._slice]
            self._pcs = unpack_tagSignature(pcsbuffer[0:4])
            self._pcsdescription = _colorspacesignatures.get(self._pcs, "None")

        except Exception:
            raise ICCFileError("file doesn't appear to have a PCS")

    @property
    def pcs(self):
        return self._pcs

    @property
    def pcsdescription(self):
        return self._pcsdescription

    def __repr__(self):
        return "<class '{0}({1}, PCS('{2}', '{3}'))'>" .format(
            self.__class__.__name__, self._slice,
            self._pcs, self._pcsdescription)

    def __str__(self):
        return "{:>30}{:5}\"{:<}\", \"{}\"".format(
            "Profile PCS:",
            "",
            str(self._pcs),
            str(self._pcsdescription)
        )


class iccDateTimeNumber(iccProfileElement):
    def __init__(self):
        super(iccDateTimeNumber, self).__init__(24, 12)
        self._datetime = None

    def read(self, buffer):
        try:
            datetimebuffer = buffer[self._slice]
            self._datetime = datetime.datetime(
                unpack_uInt16Number(datetimebuffer[0:2]),
                unpack_uInt16Number(datetimebuffer[2:4]),
                unpack_uInt16Number(datetimebuffer[4:6]),
                unpack_uInt16Number(datetimebuffer[6:8]),
                unpack_uInt16Number(datetimebuffer[8:10]),
                unpack_uInt16Number(datetimebuffer[10:12])
            )
        except ValueError as error:
            print("iccDateTimeNumber Exception: {}".format(error))
            pass
        except Exception:
            raise ICCFileError("file doesn't appear to have a datetime number")

    @property
    def datetime(self):
        return self._datetime

    def __repr__(self):
        return "<class '{0}({1}, datetime({2}))'>".format(
            self.__class__.__name__, self._slice, self._datetime
        )

    def __str__(self):
        return "{:>30}{:5}{:<}".format(
            "Date and Time Created:",
            "",
            str(self._datetime)
        )


class iccProfileFileSignature(iccProfileElement):
    def __init__(self):
        super(iccProfileFileSignature, self).__init__(36, 4)
        self._profilefilesignature = None

    @property
    def profilefilesignature(self):
        return self._profilefilesignature

    def read(self, buffer):
        try:
            profilefilesignaturebuffer = buffer[self._slice]
            self._profilefilesignature = unpack_tagSignature(
                profilefilesignaturebuffer)

            if self._profilefilesignature != "acsp":
                raise ICCFileError("file doesn't appear to be an ICC / ICM")

        except Exception:
            raise ICCFileError("file doesn't appear to have a profile file "
                               "signature")

    def __repr__(self):
        return "<class '{0}({1}, profilefilesignature({2}))'>".format(
            self.__class__.__name__, self._slice, self._profilefilesignature
        )

    def __str__(self):
        return "{:>30}{:5}\"{:<}\"".format(
            "Profile File Signature:",
            "",
            str(self._profilefilesignature)
        )


class iccPrimaryPlatform(iccProfileElement):
    def __init__(self):
        super(iccPrimaryPlatform, self).__init__(40, 4)
        self._primaryplatform = None
        self._primaryplatformdescription = None

    @property
    def primaryplatform(self):
        return self._primaryplatform

    @property
    def primaryplatformdescription(self):
        return self._primaryplatformdescription

    def read(self, buffer):
        try:
            primaryplatformbuffer = buffer[self._slice]
            self._primaryplatform = unpack_tagSignature(
                primaryplatformbuffer[0:4])
            self._primaryplatformdescription = \
                _primaryplatformsignatures.get(self._primaryplatform, "None")

        except Exception:
            raise ICCFileError(
                "file doesn't appear to have a primary platform"
            )

    def __repr__(self):
        return "<class '{0}({1}, primaryplatform('{2}', '{3}'))'>".format(
            self.__class__.__name__, self._slice, self._primaryplatform,
            self._primaryplatformdescription
        )

    def __str__(self):
        return "{:>30}{:5}\"{:<}\", \"{}\"".format(
            "Primary Platform:",
            "",
            str(self._primaryplatform),
            str(self._primaryplatformdescription)
        )


class iccProfileFlags(iccProfileElement):
    def __init__(self):
        super(iccProfileFlags, self).__init__(44, 4)
        self._profileflags = None

    @property
    def profileflags(self):
        return self._profileflags

    def read(self, buffer):
        try:
            profileflagsbuffer = buffer[self._slice]
            self._profileflags = struct.unpack(
                "=4b", profileflagsbuffer[0:4])[0]

        except Exception:
            raise ICCFileError("file doesn't appear to have profile flags")

    def __repr__(self):
        return "<class '{0}({1}, profileflags({2:032b}))'>".format(
            self.__class__.__name__, self._slice, self._profileflags
        )

    def __str__(self):
        return "{:>30}{:5}{:<}".format(
            "Profile Flags:",
            "",
            str(self._profileflags)
        )


class iccDeviceManufacturer(iccProfileElement):
    def __init__(self):
        super(iccDeviceManufacturer, self).__init__(48, 4)
        self._devicemanufacturer = None

    @property
    def devicemanufacturer(self):
        return self._devicemanufacturer

    def read(self, buffer):
        try:
            devicemanufacturerbuffer = buffer[self._slice]
            self._devicemanufacturer = unpack_tagSignature(
                devicemanufacturerbuffer[0:4])

        except Exception:
            raise ICCFileError("file doesn't appear to have a device "
                               "manufacturer")

    def __repr__(self):
        return "<class '{0}({1}, devicemanufacturer('{2}'))'>".format(
            self.__class__.__name__, self._slice, self._devicemanufacturer
        )

    def __str__(self):
        return "{:>30}{:5}\"{:<}\"".format(
            "Device Manufacturer:",
            "",
            str(self._devicemanufacturer)
        )


class iccDeviceModel(iccProfileElement):
    def __init__(self):
        super(iccDeviceModel, self).__init__(52, 4)
        self._devicemodel = None

    @property
    def devicemodel(self):
        return self._devicemodel

    def read(self, buffer):
        try:
            devicemodelbuffer = buffer[self._slice]
            self._devicemodel = unpack_tagSignature(devicemodelbuffer[0:4])

        except Exception:
            raise ICCFileError("file doesn't appear to have a device model")

    def __repr__(self):
        return "<class '{0}({1}, devicemodel('{2}'))'>".format(
            self.__class__.__name__, self._slice, self._devicemodel
        )

    def __str__(self):
        return "{:>30}{:5}\"{:<}\"".format(
            "Device Model:",
            "",
            str(self._devicemodel)
        )


class iccDeviceAttributes(iccProfileElement):
    def __init__(self):
        super(iccDeviceAttributes, self).__init__(56, 8)
        self._deviceattributes = None

    @property
    def deviceattributes(self):
        return self._deviceattributes

    def read(self, buffer):
        try:
            deviceattributesbuffer = buffer[self._slice]
            self._deviceattributes = struct.unpack(
                "=8b", deviceattributesbuffer[0:8])[0]

        except Exception:
            raise ICCFileError("file doesn't appear to have device "
                               "attributes flags"
                               )

    def __repr__(self):
        return "<class '{0}({1}, deviceattributesflags({2:064b}))'>".format(
            self.__class__.__name__, self._slice, self._deviceattributes
        )

    def __str__(self):
        return "{:>30}{:5}{:<b}".format(
            "Device Attributes:",
            "",
            self._deviceattributes
        )


class iccRenderingIntent(iccProfileElement):
    def __init__(self):
        super(iccRenderingIntent, self).__init__(64, 4)
        self._renderingintent = None
        self._renderingintentdescription = None

    @property
    def renderingintent(self):
        return self._renderingintent

    @property
    def renderinintentdescription(self):
        return self._renderingintentdescription

    def read(self, buffer):
        try:
            renderingintentbuffer = buffer[self._slice]
            self._renderingintent = unpack_uInt32Number(
                renderingintentbuffer[0:4])
            self._renderingintentdescription = \
                _renderingintenttable[self._renderingintent]

        except Exception:
            raise ICCFileError("file doesn't appear to have a rendering intent"
                               )

    def __repr__(self):
        return "<class '{0}({1}, renderingintent({2}, '{3}'))'>".format(
            self.__class__.__name__, self._slice, self._renderingintent,
            self._renderingintentdescription
        )

    def __str__(self):
        return "{:>30}{:5}{:<}, \"{}\"".format(
            "Profile PCS:",
            "",
            self._renderingintent,
            str(self._renderingintentdescription)
        )


class iccPCSIlluminant(iccProfileElement):
    def __init__(self):
        super(iccPCSIlluminant, self).__init__(68, 12)
        self._pcsilluminant = XYZNumber()

    @property
    def pcsilluminant(self):
        return self._pcsilluminant

    def read(self, buffer):
        try:
            pcsilluminantxyzbuffer = buffer[self._slice]
            self._pcsilluminant.read(pcsilluminantxyzbuffer)

        except Exception:
            raise ICCFileError(
                "file doesn't appear to have a rendering intent")

    def __repr__(self):
        return "<class '{0}({1}, pcsilluminant({2}))'>".format(
            self.__class__.__name__, self._slice, repr(self._pcsilluminant))

    def __str__(self):
        return "{:>30}{:5}{:<}".format(
            "PCS Illuminant:",
            "",
            str(self._pcsilluminant)
        )


class iccProfileCreator(iccProfileElement):
    def __init__(self):
        super(iccProfileCreator, self).__init__(80, 4)
        self._profilecreator = None

    @property
    def profilecreator(self):
        return self._profilecreator

    def read(self, buffer):
        try:
            profilecreatorbuffer = buffer[self._slice]
            self._profilecreator = unpack_tagSignature(
                profilecreatorbuffer[0:4])

        except Exception:
            raise ICCFileError("file doesn't appear to have a profile creator")

    def __repr__(self):
        return "<class '{0}({1}, profilecreator('{2}'))'>".format(
            self.__class__.__name__, self._slice, self._profilecreator
        )

    def __str__(self):
        return "{:>30}{:5}\"{:<}\"".format(
            "Profile Creator:",
            "",
            str(self._profilecreator)
        )


class iccProfileID(iccProfileElement):
    def __init__(self):
        super(iccProfileID, self).__init__(84, 16)
        self._profileid = bytes(16)

    @property
    def profileid(self):
        return self._profileid

    def read(self, buffer):
        try:
            profileidbuffer = buffer[self._slice]
            count = len(profileidbuffer)
            self._profileid = struct.unpack(
                "{0}s".format(count), profileidbuffer)[0]

        except Exception:
            raise ICCFileError("file doesn't appear to have a profile ID")

    def __repr__(self):
        return "<class '{0}({1}, profileid('{2}'))'>".format(
            self.__class__.__name__, self._slice, self._profileid.hex()
        )

    def __str__(self):
        return "{:>30}{:5}{:<}".format(
            "Profile ID:",
            "",
            str(self._profileid.hex())
        )


class iccReserved(iccProfileElement):
    def __init__(self):
        super(iccReserved, self).__init__(100, 28)
        self._reserved = bytes(28)

    def read(self, buffer):
        try:
            reservedbuffer = buffer[self._slice]
            count = len(reservedbuffer)
            self._reserved = struct.unpack(
                "{0}s".format(count), reservedbuffer)[0]

        except Exception:
            raise ICCFileError("file doesn't appear to have a reserved field")

    def __repr__(self):
        return "<class '{0}({1}, reserved('{2}'))'>".format(
            self.__class__.__name__, self._slice, self._reserved.hex()
        )

    def __str__(self):
        return "{:>30}{:5}{:<}".format(
            "Reserved:",
            "",
            str(self._reserved.hex())
        )


class iccTag(iccProfileElement):
    def __init__(self, offset):
        super(iccTag, self).__init__(offset, 12)
        self._tagsignature = None
        self._tagoffset = None
        self._tagsize = None
        self._type = None

    @property
    def signature(self):
        return self._tagsignature

    @property
    def type(self):
        return self._type

    def read(self, buffer):
        try:
            tagbuffer = buffer[self._slice]
            self._tagsignature = unpack_tagSignature(tagbuffer[0:4])
            self._tagoffset = unpack_uInt32Number(tagbuffer[4:8])
            self._tagsize = unpack_uInt32Number(tagbuffer[8:12])

            try:
                signaturetype = unpack_tagSignature(
                    buffer[
                        self._tagoffset:self._tagoffset + 4
                    ]
                )

                signaturetype = signaturetype.replace(" ", "_")
                signatureclass = getattr(
                    sys.modules[__name__],
                    "{}Type".format(signaturetype)
                )

                self._type = signatureclass(
                    self._tagoffset,
                    self._tagsize,
                    buffer
                )

            except AttributeError:
                pass

        except Exception:
            raise ICCFileError("error while reading tag signature")

    def __repr__(self):
        return "<class '{0}({1}, tag('{2}'))'>".format(
            self.__class__.__name__, self._slice, self._tagsignature
        )

    def __str__(self):
        return "{} Offset: {} Size: {} {}".format(
            "{:<8}".format("\"" + str(self._tagsignature) + "\","),
            "{:<8}".format(str(self._tagoffset) + ","),
            "{:<8}".format(str(self._tagsize) + ","),
            str(self._type)
        )


class iccTagTable(iccProfileElement):
    def __init__(self):
        super(iccTagTable, self).__init__(128, 4)
        self._tagcount = None
        self._tags = None

    @property
    def count(self):
        return self._tagcount

    @property
    def tags(self):
        return self._tags

    def read(self, buffer):
        try:
            tagtablebuffer = buffer[self._slice]
            self._tagcount = unpack_uInt32Number(tagtablebuffer)

            self._tags = numpy.empty(
                self._tagcount,
                dtype=[("signature", "U4"), ("tag", object)]
            )

            for count in range(self._tagcount):
                tag = iccTag((count * 12) + 132)
                tag.read(buffer)
                self._tags[count] = (tag.signature, tag)

        except Exception:
            raise ICCFileError("file doesn't appear to have a tag table")

    def __repr__(self):
        return "<class '{0}({1}, tagtable('{2}'))'>".format(
            self.__class__.__name__, self._slice, self._tagcount
        )

    def __str__(self):
        tagstring = ""
        for index, tag in numpy.ndenumerate(self._tags):
            if index[0] == 0:
                tagstring += str(tag["tag"])
            else:
                tagstring += "\n{:35}{:<}".format(
                    "",
                    str(tag["tag"])
                )
        return "{:>30}{:5}{:<}\n{:>30}{:5}{:10}".format(
            "Tag Table Count:",
            "",
            str(self._tagcount),
            "Tag Signatures:",
            "",
            tagstring
        )


class iccProfile:
    def __init__(self):
        self._profilesize = iccProfileSize()
        self._preferredcmmtype = iccPreferredCMMType()
        self._profileversion = iccProfileVersion()
        self._profiledeviceclass = iccProfileDeviceClass()
        self._datacolorspace = iccDataColorSpace()
        self._pcs = iccPCS()
        self._datetimenumber = iccDateTimeNumber()
        self._profilefilesignature = iccProfileFileSignature()
        self._primaryplatform = iccPrimaryPlatform()
        self._profileflags = iccProfileFlags()
        self._devicemanufacturer = iccDeviceManufacturer()
        self._devicemodel = iccDeviceModel()
        self._deviceattributes = iccDeviceAttributes()
        self._renderingintent = iccRenderingIntent()
        self._pcsilluminant = iccPCSIlluminant()
        self._profilecreator = iccProfileCreator()
        self._profileid = iccProfileID()
        self._reserved = iccReserved()
        self._tagtable = iccTagTable()

    @property
    def tags(self):
        return self._tagtable.tags

    def read(self, buffer):
        for _, var in vars(self).items():
            try:
                var.read(buffer)
            except Exception as e:
                print("Skip {}: {}".format(type(var).__name__, str(e)))
                continue

    def __str__(self):
        string = ""
        append = False

        for _, var in vars(self).items():
            if append is True:
                string += "\n"
            else:
                append = True

            string += str(var)
        return string


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            prog='iccinspector'
        )
        parser.add_argument(
            "iccfile",
            type=argparse.FileType("rb")
        )
        parser.add_argument(
            "-t",
            dest="tagsignature",
            action="append",
            help="specify tag signature to be inspected"
        )
        parser.add_argument(
            "-e",
            dest="extract_lut",
            action="store_true",
            help="extract lut from their respective tags"
        )
        args = parser.parse_args()

        numpy.set_printoptions(15, threshold=128)

        with args.iccfile as f:
            s = memoryview(f.read())

            testField = iccProfile()
            testField.read(s)
            print(testField)

            # for tagsignature in args.tagsignature:
            #     print(
            #         testField.tags[
            #             numpy.where(testField.tags["signature"] ==
            #             tagsignature)
            #         ]
            #     )

        if args.extract_lut:
            for (signature, tag) in testField.tags:
                try:
                    tag.type.extract_lut(signature)
                except Exception:
                    continue

    except ICCFileError as e:
        print("Error loading ICC / ICM file: {}".format(e))
    except Exception as e:
        raise
