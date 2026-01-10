## A mapping of BookNLP Keywords

|Source Keyword|Source File|Target Keyword|Target File|Description|
|:--|:--|:--|:--|:--|
|`agent > i`|`.book`|`token_ID_within_document`|`.tokens`| Corresponds to the token index within the document|
|`characters > id`|`.book`|`COREF`|`.entities`| Corresponds to the ID of an extracted entity |
|`mentions > proper > c`|`.book`|—|—| Corresponds to the number of times/count an entity was referred to by a specific proper name |
|`sentence_ID`|`.tokens`|—|—| Corresponds to the ID of a sentence within the processed document |
|`events`|`.tokens`|—|—| Indicates whether the token was identified to have triggered an event via `events == EVENT` |
