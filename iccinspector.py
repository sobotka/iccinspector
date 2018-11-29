#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import struct
import sys
import os
import datetime

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
    return numpy.divide(
        numpy.frombuffer(s, numpy.dtype(">i4"), len(s) // 4), 2**16)


def unpack_tagSignature(s):
    """Convert sequence of 4 bytes into a string."""
    return struct.unpack("=4s", s)[0].decode("utf-8")


def unpack_uInt32Number(s):
    """Convert sequence of 4 bytes into a big endian 32 bit unsigned int."""
    return struct.unpack(">I", s)[0]


def unpack_uInt16Number(s):
    """Convert sequence of 2 bytes into a big endian 16 bit unsigned int."""
    return struct.unpack(">H", s)[0]


class XYZNumber:
    def __init__(self):
        self._XYZ = numpy.array([0., 0., 0.])

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

    def read(self, buffer):
        try:
            self._XYZ = unpack_s15Fixed16Number(buffer)

        except Exception:
            raise ICCFileError("file doesn't appear to have a profile size")

    def __repr__(self):
        return "<class '{0}(XYZ({1}))'>".format(
            self.__class__.__name__, self._XYZ)

    def __str__(self):
        return "X: {0:<015}, Y: {1:<015}, Z:{2:<015}".format(
            float(self._XYZ[0]), float(self._XYZ[1]), float(self._XYZ[2]))


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
            raise ICCFileError("file doesn't appear to have profile flags"
                               )

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

            if (None):
                raise Exception

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

    def read(self, buffer):
        for _, var in vars(self).items():
            var.read(buffer)

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
        numpy.set_printoptions(15)

        with open(sys.argv[1], 'rb') as f:
            s = memoryview(f.read())

            testField = iccProfile()
            testField.read(s)
            print(testField)

    except ICCFileError as e:
        print("Error encountered when attempting to load the ICC / ICM. "
              "Error:", e)
    except FileNotFoundError as e:
        print("Unable to find file. Error:", e)
    except IndexError as e:
        print("Usage: iccinspector iccfile", e)
    except Exception as e:
        raise
