# pip install huaweicloudsdkcore huaweicloudsdkfrs
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkfrs.v2.region.frs_region import FrsRegion
from huaweicloudsdkfrs.v2 import *
import base64
import json

def encode_base64(file):
    with open(file,'rb') as f:
        img_data = f.read()
        base64_data = str(base64.b64encode(img_data), 'utf-8')  
        return base64_data

def GetClient():
    return FrsClient.new_builder(FrsClient) \
         .with_credentials(BasicCredentials("XMXPEGM2LMNC8EBXJRZJ", "x1X6M0zrNq3YBNMzOZery81xm1cSV7TpUX2uSiUf")) \
         .with_region(FrsRegion.value_of("cn-north-4")) \
         .build()

def searchFaceByBase64(client, imageFileName):
    try:
        request = SearchFaceByBase64Request()
        request.face_set_name = "library"
        # listFaceSearchBase64ReqReturnFieldsbody = [
        #     "timestamp"
        # ]

        request.body = FaceSearchBase64Req(
            # return_fields=listFaceSearchBase64ReqReturnFieldsbody,
            image_base64=encode_base64(imageFileName)
        )
        response = client.search_face_by_base64(request)
        # print(response)
        for i in json.loads(str(response))['faces']:
            print(i['external_image_id'],i["similarity"])
    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)

def addFacesByBase64(client, imageFileName, imageLabel):
    try:
        request = AddFacesByBase64Request()
        request.face_set_name = "library"
        request.body = AddFacesBase64Req(
            # external_fields="{\"timestamp\":12}",
            external_image_id=imageLabel,
            image_base64=encode_base64(imageFileName)
        )
        response = client.add_faces_by_base64(request)
        print(response)
    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)

client = GetClient()
# searchFaceByBase64(client,"fbb-1.jpg")

# addFacesByBase64(client,"zly-1.jpg","zly")
# searchFaceByBase64(client,"zly-2.jpg")

# addFacesByBase64(client,"zys-1.jpg","zys")
# addFacesByBase64(client,"zys-2.jpg","zys")
searchFaceByBase64(client,"zys-3.jpg")
