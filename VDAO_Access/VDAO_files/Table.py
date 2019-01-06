# The modeling here makes a "Table belong to a video", not the opposite.
# It may sound awkward, but modeling it this way makes the search for specific videos easier.
# All the filters are applied in the lowest level object: VideoFile

import json

import _init_paths
from VDAOHelper import VideoType


class SourcePackage:
    @staticmethod
    def CreateSourcePackage(jsonVideos):
        sourcePackage = SourcePackage()
        _videos = []
        _tables = []
        for tag in jsonVideos:
            if tag == "source":
                sourcePackage.SetSource(jsonVideos[tag])
            elif tag == "description":
                sourcePackage.SetDescrition(jsonVideos[tag])
            elif tag == "tables":
                tables = jsonVideos[tag]
                for tableName in tables:
                    [t, r, o] = Table.CreateTable(tables[tableName], tableName, sourcePackage)
                    _videos = _videos + r + o
                    _tables.append(t)
        return [sourcePackage, _tables, _videos]

    def SetSource(self, source):
        self.source = source

    def SetDescrition(self, description):
        self.description = description


class Table:
    @staticmethod
    def CreateTable(jsonTable, tableName, sourceVideos):
        table = Table()
        _references = []
        _objects = []
        table.name = tableName
        if "description" in jsonTable:
            table.SetDescription(jsonTable["description"])
        if "part" in jsonTable:
            table.SetPart(jsonTable["part"])
        if "type" in jsonTable:
            table.SetType(jsonTable["type"])
        if "illumination" in jsonTable:
            table.SetIllumination(jsonTable["illumination"])
        if "references" in jsonTable:
            references = jsonTable["references"]
            for reference in references:
                _references.append(VideoFile(reference, VideoType.Reference, table, sourceVideos))
        if "objects" in jsonTable:
            objects = jsonTable["objects"]
            for objs in objects:
                _objects.append(VideoFile(objs, VideoType.WithObjects, table, sourceVideos))
        return [table, _references, _objects]

    def SetDescription(self, description):
        self.description = description

    def SetPart(self, part):
        self.part = part

    def SetType(self, type):
        self.type = type

    def SetIllumination(self, illumination):
        self.illumination = illumination


class VideoFile:
    def __init__(self, jsonVideo, videoType, sourceTable, sourceVideos):
        self.sourceTable = sourceTable
        self.sourceVideos = sourceVideos
        self.videoType = videoType
        if "name" in jsonVideo:
            self.name = jsonVideo["name"]
        if "object_class" in jsonVideo:
            self.object_class = jsonVideo["object_class"]
        if "position" in jsonVideo:
            self.position = jsonVideo["position"]
        if "part" in jsonVideo:
            self.part = jsonVideo["part"]
        if "video" in jsonVideo:
            self.video = jsonVideo["video"]
        if "version" in jsonVideo:
            self.version = jsonVideo["version"]
        if "url" in jsonVideo:
            self.url = jsonVideo["url"]
        if "url_annotation" in jsonVideo:
            self.url_annotation = jsonVideo["url_annotation"]


class Filters:
    def __init__(self, sourcePackage, tables, videos):
        self.sourcePackage = sourcePackage
        self.tables = tables
        self.videos = videos

    def GetAllTableNames(self):
        ret = []
        for t in self.tables:
            if t.name not in ret:
                ret.append(t.name)
        return ret

    def GetAllObjectsClasses(self):
        ret = []
        for o in self.videos:
            if (o.videoType == VideoType.WithObjects and o.object_class not in ret):
                ret.append(o.object_class)
        # if 'multiple objects' in ret:
        #     ret.remove('multiple objects')
        return ret

    def GetAllTypes(self):
        ret = []
        for o in self.tables:
            if o.type not in ret:
                ret.append(o.type)
        return ret

    def GetIlluminationTypes(self):
        ret = []
        for o in self.tables:
            if hasattr(o, 'illumination'):
                if o.illumination not in ret:
                    ret.append(o.illumination)
        return ret

    def GetVideosFromTable(self, tableName, objectClasses=None):
        ret = []
        for vid in self.videos:
            if (vid.videoType == VideoType.WithObjects and vid.sourceTable.name == tableName):
                if objectClasses != None:
                    [ret.append(vid) for ocf in objectClasses
                     if vid.object_class == ocf]  #if object must be found
                else:
                    ret.append(vid)
        return ret

    #tables = "table 1", "table 2" ...
    #objects = "shoes", "dark-blue box" ...
    #videoTypes = "references" or "with objects"
    #illuminations = "normal" or "extra"
    def GetVideos(self, tables=None, objects=None, videoTypes=None, illuminations=None):
        filteredTables = []
        filteredIlluminations = []
        # Get videos
        for vid in self.videos:
            # Get all videos by required tables
            if tables is not None:
                for table in tables:
                    if vid.sourceTable.name == table:
                        filteredTables.append(vid)
        # Filter by tables that have required illuminations
        if illuminations is not None:
            for vid in filteredTables:
                for illum in illuminations:
                    if vid.sourceTable.illumination == illum:
                        filteredIlluminations.append(vid)
        # If reference videos are not required, remove them from the illumination
        vids2Remove = []
        if ('reference' not in [v.lower() for v in videoTypes]):
            for vid in filteredIlluminations:
                if vid.videoType == VideoType.Reference:
                    vids2Remove.append(vid.url)  #identify vids to remove by its url
        for vid2remove in vids2Remove:
            for vid in filteredIlluminations:
                if vid.url == vid2remove:
                    filteredIlluminations.remove(vid)
                    break
        # If videos with objects are not required, remove them from the illumination
        vids2Remove = []
        if ('with objects' not in [v.lower() for v in videoTypes]):
            for vid in filteredIlluminations:
                if vid.videoType == VideoType.WithObjects:
                    vids2Remove.append(vid.url)  #identify vids to remove by its url
        for vid2remove in vids2Remove:
            for vid in filteredIlluminations:
                if vid.url == vid2remove:
                    filteredIlluminations.remove(vid)
                    break
        # Remove videos of objects that are not in the list objects
        vids2Remove = []
        for vid in filteredIlluminations:
            if vid.videoType == VideoType.WithObjects:
                if vid.object_class not in objects:
                    vids2Remove.append(vid.url)  #identify vids to remove by its url
        for vid2remove in vids2Remove:
            for vid in filteredIlluminations:
                if vid.url == vid2remove:
                    filteredIlluminations.remove(vid)
                    break
        return filteredIlluminations
