from __future__ import annotations
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Ancient Language Data Service", page_icon="🏛️", layout="wide", initial_sidebar_state="collapsed"
)

# ## Set up MACULA dataframes
verse_df = pd.read_csv("databases/preprocessed-macula-dataframes/verse_df.csv")
mg = pd.read_csv("databases/preprocessed-macula-dataframes/mg.csv")
# mh = pd.read_csv("preprocessed-macula-dataframes/mh.csv")


from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool, tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Set up the content from a few modules

# """Callback Handler captures all callbacks in a session for future offline playback."""


import pickle
import time
from typing import Any, TypedDict

from langchain.callbacks.base import BaseCallbackHandler


# This is intentionally not an enum so that we avoid serializing a
# custom class with pickle.
class CallbackType:
    ON_LLM_START = "on_llm_start"
    ON_LLM_NEW_TOKEN = "on_llm_new_token"
    ON_LLM_END = "on_llm_end"
    ON_LLM_ERROR = "on_llm_error"
    ON_TOOL_START = "on_tool_start"
    ON_TOOL_END = "on_tool_end"
    ON_TOOL_ERROR = "on_tool_error"
    ON_TEXT = "on_text"
    ON_CHAIN_START = "on_chain_start"
    ON_CHAIN_END = "on_chain_end"
    ON_CHAIN_ERROR = "on_chain_error"
    ON_AGENT_ACTION = "on_agent_action"
    ON_AGENT_FINISH = "on_agent_finish"


# We use TypedDict, rather than NamedTuple, so that we avoid serializing a
# custom class with pickle. All of this class's members should be basic Python types.
class CallbackRecord(TypedDict):
    callback_type: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    time_delta: float  # Number of seconds between this record and the previous one


def load_records_from_file(path: str) -> list[CallbackRecord]:
    """Load the list of CallbackRecords from a pickle file at the given path."""
    with open(path, "rb") as file:
        records = pickle.load(file)

    if not isinstance(records, list):
        raise RuntimeError(f"Bad CallbackRecord data in {path}")
    return records


def playback_callbacks(
    handlers: list[BaseCallbackHandler],
    records_or_filename: list[CallbackRecord] | str,
    max_pause_time: float,
) -> str:
    if isinstance(records_or_filename, list):
        records = records_or_filename
    else:
        records = load_records_from_file(records_or_filename)

    for record in records:
        pause_time = min(record["time_delta"], max_pause_time)
        if pause_time > 0:
            time.sleep(pause_time)

        for handler in handlers:
            if record["callback_type"] == CallbackType.ON_LLM_START:
                handler.on_llm_start(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_LLM_NEW_TOKEN:
                handler.on_llm_new_token(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_LLM_END:
                handler.on_llm_end(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_LLM_ERROR:
                handler.on_llm_error(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TOOL_START:
                handler.on_tool_start(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TOOL_END:
                handler.on_tool_end(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TOOL_ERROR:
                handler.on_tool_error(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TEXT:
                handler.on_text(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_CHAIN_START:
                handler.on_chain_start(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_CHAIN_END:
                handler.on_chain_end(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_CHAIN_ERROR:
                handler.on_chain_error(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_AGENT_ACTION:
                handler.on_agent_action(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_AGENT_FINISH:
                handler.on_agent_finish(*record["args"], **record["kwargs"])

    # Return the agent's result
    for record in records:
        if record["callback_type"] == CallbackType.ON_AGENT_FINISH:
            return record["args"][0][0]["output"]

    return "[Missing Agent Result]"


class CapturingCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._records: list[CallbackRecord] = []
        self._last_time: float | None = None

    def dump_records_to_file(self, path: str) -> None:
        """Write the list of CallbackRecords to a pickle file at the given path."""
        with open(path, "wb") as file:
            pickle.dump(self._records, file)

    def _append_record(
        self, type: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        time_now = time.time()
        time_delta = time_now - self._last_time if self._last_time is not None else 0
        self._last_time = time_now
        self._records.append(
            CallbackRecord(
                callback_type=type, args=args, kwargs=kwargs, time_delta=time_delta
            )
        )

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_START, args, kwargs)

    def on_llm_new_token(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_NEW_TOKEN, args, kwargs)

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_END, args, kwargs)

    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_ERROR, args, kwargs)

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TOOL_START, args, kwargs)

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TOOL_END, args, kwargs)

    def on_tool_error(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TOOL_ERROR, args, kwargs)

    def on_text(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TEXT, args, kwargs)

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_CHAIN_START, args, kwargs)

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_CHAIN_END, args, kwargs)

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_CHAIN_ERROR, args, kwargs)

    def on_agent_action(self, *args: Any, **kwargs: Any) -> Any:
        self._append_record(CallbackType.ON_AGENT_ACTION, args, kwargs)

    def on_agent_finish(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_AGENT_FINISH, args, kwargs)


# A hack to "clear" the previous result when submitting a new prompt. This avoids
# the "previous run's text is grayed-out but visible during rerun" Streamlit behavior.
class DirtyState:
    NOT_DIRTY = "NOT_DIRTY"
    DIRTY = "DIRTY"
    UNHANDLED_SUBMIT = "UNHANDLED_SUBMIT"


def get_dirty_state() -> str:
    return st.session_state.get("dirty_state", DirtyState.NOT_DIRTY)


def set_dirty_state(state: str) -> None:
    st.session_state["dirty_state"] = state


def with_clear_container(submit_clicked: bool) -> bool:
    if get_dirty_state() == DirtyState.DIRTY:
        if submit_clicked:
            set_dirty_state(DirtyState.UNHANDLED_SUBMIT)
            st.experimental_rerun()
        else:
            set_dirty_state(DirtyState.NOT_DIRTY)

    if submit_clicked or get_dirty_state() == DirtyState.UNHANDLED_SUBMIT:
        set_dirty_state(DirtyState.DIRTY)
        return True

    return False

# # Expand functionality for more tools using DB lookups

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

# bible_persist_directory = '/Users/ryderwishart/genesis/databases/berean-bible-database'
bible_persist_directory = "/Users/ryderwishart/genesis/databases/berean-bible-database"
bible_chroma = Chroma(
    "berean-bible", embeddings, persist_directory=bible_persist_directory
)
# print(bible_chroma.search("jesus speaks to peter", search_type="similarity", k=1))

encyclopedic_persist_directory = "/Users/ryderwishart/biblical-machine-learning/gpt-inferences/databases/encyclopedic"
encyclopedic_chroma = Chroma(
    persist_directory=encyclopedic_persist_directory,
    embedding_function=embeddings,
    collection_name="encyclopedic",
)
# print(
#     encyclopedic_chroma.similarity_search_with_score(
#         "What is a sarcophagus?", search_type="similarity", k=1
#     )
# )

theology_persist_directory = (
    "/Users/ryderwishart/biblical-machine-learning/gpt-inferences/databases/theology"
)
theology_chroma = Chroma(
    "theology", embeddings, persist_directory=theology_persist_directory
)
# print(theology_chroma.search("jesus speaks to peter", search_type="similarity", k=1))

# # persist_directory = '/Users/ryderwishart/genesis/databases/itemized-prose-contexts copy' # NOTE: Itemized prose contexts are in this db
# persist_directory = '/Users/ryderwishart/genesis/databases/prose-contexts' # NOTE: Full prose contexts are in this db
# context_chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="prosaic_contexts_itemized")
# print(context_chroma.similarity_search_with_score('jesus (s) speaks (v) to peter (o)', search_type='similarity', k=1))

persist_directory = (
    "/Users/ryderwishart/genesis/databases/prose-contexts-shorter-itemized"
)
context_chroma = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="prosaic_contexts_shorter_itemized",
)


DB_PATH = (Path(__file__).parent / "Chinook.db").absolute()

SAVED_SESSIONS = {
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?": "leo.pickle",
    "What is the full name of the artist who recently released an album called "
    "'The Storm Before the Calm' and are they in the FooBar database? If so, what albums of theirs "
    "are in the FooBar database?": "alanis.pickle",
}


"# 🏛️📚 Ancient Language Data Service"

# Setup credentials in Streamlit
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to run your own custom questions."
)

if user_openai_api_key:
    openai_api_key = user_openai_api_key
    enable_custom = True
else:
    openai_api_key = "not_supplied"
    enable_custom = False

# Tools setup
# llm = OpenAI(temperature=0, openai_api_key=openai_api_key, streaming=True)

# search = DuckDuckGoSearchAPIWrapper()
# llm_math_chain = LLMMathChain.from_llm(llm)
# db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
# db_chain = SQLDatabaseChain.from_llm(llm, db)



llm = OpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0,
    streaming=True,
)

from langchain.agents import create_pandas_dataframe_agent

macula_greek_verse_agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0),
    # mg, # verse_df (?)
    verse_df,
    # verbose=True,
)

macula_greek_words_agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0),
    # mg, # verse_df (?)
    mg,
    # verbose=True,
)

# Improve the linguistic data lookup tool with discourse feature definitions
discourse_types = {
    'Main clauses': {'description': 'Main clauses are the top-level clauses in a sentence. They are the clauses that are not embedded in other clauses.'},
    'Historical Perfect': {'description': 'Highlights not the speech or act to which it refers but the event(s) that follow (DFNTG §12.2).'},
    'Specific Circumstance': {'description': 'The function of ἐγενετο ‘it came about’ and an immediately following temporal expression varies with the author (see DFNTG §10.3). In Matthew’s Gospel, it usually marks major divisions in the book (e.g. Mt 7:28). In Luke-Acts, in contrast, ‘it picks out from the general background the specific circumstance for the foreground events that are to follow’ (ibid.), as in Acts 9:37 (see also Mt 9:10).'},
    'Verb Focus+': {'description': 'Verb in final position in clause demonstrates verb focus.'},
    'Articular Pronoun': {'description': 'Articular pronoun, which often introduces an ‘intermediate step’ in a reported conversation.'},
    'Topical Genitive': {'description': 'A genitival constituent that is nominal is preposed within the noun phrase for two purposes: 1) to bring it into focus; 2) within a point of departure, to indicate that it is the genitive in particular which relates to a corresponding constituent of the context.(DFNTG §4.5)'},
    'Embedded DFE': {'description': "'Dominant focal elements' embedded within a constituent in P1."},
    'Reported Speech': {'description': 'Reported speech.'},
    'Ambiguous': {'description': 'Marked but ambiguous constituent order.'},
    'Over-encoding': {'description': 'Any instance in which more encoding than the default is employed to refer to an active participant or prop. Over-encoding is used in Greek, as in other languages: to mark the beginning of a narrative unit (e.g. Mt 4:5); and to highlight the action or speech concerned (e.g. Mt 4:7).'},
    'Highlighter': {'description': 'Presentatives - Interjections such as ἰδού and ἴδε ‘look!, see!’ typically highlight what immediately follows (Narr §5.4.2, NonNarr §7.7.3).'},
    'Referential PoD': {'description': 'Pre-verbal topical subject other referential point of departure (NARR §3.1, NonNarr §4.3, DFNTG §§2.2, 2.8; as in 1 Th 1:6).'},
    'annotations': {'description': 'Inline annotations.'},
    'Left-Dislocation': {'description': 'Point of departure - A type of SENTENCE in which one of the CONSTITUENTS appears in INITIAL position and its CANONICAL position is filled by a PRONOUN or a full LEXICAL NOUN PHRASE with the same REFERENCE, e.g. John, I like him/the old chap.”'},
    'Focus+': {'description': 'Constituents placed in P2 to give them focal prominence.'},
    'Tail-Head linkage': {'description': 'Point of departure involving renewal - Tail-head linkage involves “the repetition in a subordinate clause, at the beginning (the ‘head’) of a new sentence, of at least the main verb of the previous sentence (the tail)” (Dooley & Levinsohn 2001:16).'},
    'Postposed them subject': {'description': 'When a subject is postposed to the end of its clause (following nominals or adjuncts), it is marked ThS+ (e.g. Lk 1:41 [twice]). Such postposing typically marks as salient the participant who performs the next event in chronological sequence in the story (see Levinsohn 2014).'},
    'EmbeddedRepSpeech': {'description': 'Embedded reported speech - speech that is reported within a reported speech.'},
    'Futuristic Present': {'description': 'Highlights not the speech or act to which it refers but the event(s) that follow (DFNTG §12.2).'},
    'OT quotes': {'description': 'Old Testament quotations.'},
    'Constituent Negation': {'description': 'Negative pro-forms when they are in P2 indicate that the constituent has been negated rather than the clause as a whole.'},
    'Split Focal': {'description': 'The second part of a focal constituent with only the first part in P2 (NonNarr §5.5, DFNTG §4.4).'},
    'Right-Dislocated': {'description': 'Point of departure - A type of SENTENCE in which one of the CONSTITUENTS appears in FINAL position and its CANONICAL position is filled by a PRONOUN with the same REFERENCE, e.g. ... He’s always late, that chap.'},
    'Appositive': {'description': 'Appositive'},
    'Situational PoD': {'description': 'Situational point of departure (e.g. temporal, spatial, conditional―(NARR §3.1, NonNarr §4.3, DFNTG §§2.2, 2.8; as in 1 Th 3:4).'},
    'Historical Present': {'description': 'Highlights not the speech or act to which it refers but the event(s) that follow (DFNTG §12.2).'},
    'Noun Incorporation': {'description': 'Some nominal objects that appear to be in P2 may precede their verb because they have been “incorporated” (Rosen 1989) in the verb phrase. Typically, the phrase consists of an indefinite noun and a “light verb” such as “do, give, have, make, take” (Wikipedia entry on Light Verbs).'},
    'Thematic Prominence': {'description': 'Thematic prominence - In Greek, prominence is given to active participants and props who are the current centre of attention (NARR §4.6) by omitting the article (DFNTG §§9.2.3-9.4), by adding αυτος ‘-self’ (e.g. in 1 Th 3:11), by using the proximal demonstrative οὗτος (NARR chap. 9, Appendix 1; e.g. in 3:3), and by postposing the constituent concerned (e.g. Mt 14:29). If such constituents are NOT in postion P1, they are demonstrating topical prominence.'},
    'Cataphoric Focus': {'description': 'An expression that points forward to and highlights something which ‘is about to be expressed.’'},
    'Cataphoric referent': {'description': 'The clause or sentence to which a cataphoric reference refers when NOT introduced with ὅτι or ἵνα.'},
    'DFE': {'description': 'Constituents that may be moved from their default position to the end of a proposition to give them focal prominence include verbs, pronominals and objects that follow adjuncts (NonNarr §5.3, DFNTG §3.5). Such constituents, also called ‘dominant focal elements’or DFEs (Heimedinger 1999:167).'},
    'Embedded Focus+': {'description': 'A constituent of a phrase or embedded clause preposed for focal prominence.'}
}

@tool
def linguistic_data_lookup_tool(query):
    """Query the linguistic data for relevant documents and add explanatory suffix if appropriate."""
    context_docs = context_chroma.similarity_search(query, k=3)
    explanatory_suffix = "Here are the definitions of the relevant discourse features:"
    include_suffix_flag = False
    for discourse_type in discourse_types.keys():
        if discourse_type in query:
            explanatory_suffix += f"\n\n{discourse_type}: {discourse_types[discourse_type]['description']}"
            include_suffix_flag = True
    if include_suffix_flag:
        context_docs.append(explanatory_suffix)
    return str(context_docs)
            

tools = [
    Tool(
        name="Bible Verse Reader Lookup",
        # Use the
        # func=lambda x: bible_chroma.search(x, search_type="similarity", k=2),
        # func=lambda x: bible_tool({"question": x}, return_only_outputs=True),
        # func=lambda x: get_relevant_bible_verses(x),
        func=lambda x: bible_chroma.search(x, search_type="similarity", k=10),
        description="useful for finding verses that are similar to the user's query, not suitable for complex queries. Be very careful to check whether the verses are actually relevant to the user's question and not just similar to the user's question in superficial ways",
        # callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    Tool(
        name="Bible Words Lookup",
        func=macula_greek_words_agent.run,  # Note: using the NT-only agent here
        description="useful for finding information about individual biblical words from a Greek words dataframe, which includes glosses, lemmas, normalized forms, and more. This tool is not useful for grammar and syntax questions (about subjects, objects, verbs, etc.), but is useful for finding information about the words themselves",
    ),
    Tool(
        name="Bible Verse Dataframe Tool",
        func=macula_greek_verse_agent.run,  # Note: using the NT-only agent here
        description="useful for finding information about Bible verses in a bible verse dataframe in case counting, grouping, aggregating, or list building is required. This tool is not useful for grammar and syntax questions (about subjects, objects, verbs, etc.), but is useful for finding information about the verses (English or Greek or Greek lemmas) themselves",
        # callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    Tool(
        name="Linguistic Data Lookup",
        # func=lambda x: context_chroma.similarity_search(x, k=3),
        func=linguistic_data_lookup_tool.run,
        description="useful for finding answers about linguistics, discourse, situational context, participants, semantic roles (source/agent, process, goal, etc.), or who the speakers are in a passage. Input MUST ALWAYS include a scope keyword like 'discourse', 'roles', or 'situation'",
    ),
    # Tool(
    #     name="Context for Most Relevant Passage", # NOTE: this tool isn't working quite right. Needs some work
    #     func=get_context_for_most_relevant_passage.run,
    #     description="useful for when you need to find relevant linguistic context for a Bible passage. Input should be 'situation for' and the original user query",
    #     callbacks=[StreamlitSidebarCallbackHandler()],
    # ),
    Tool(
        name="Syntax Data Lookup",
        func=lambda x: get_syntax_for_query(x),
        description="useful for finding syntax data about the user's query. Use this if the user is asking a question that relates to a sentence's structure, such as 'who is the subject of this sentence?' or 'what are the circumstances of this verb?'",
        # callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    Tool(
        name="Theological Data Lookup",
        func=lambda x: theology_chroma.search(x, search_type="similarity", k=5),
        # func=lambda query: get_similar_resource(theology_chroma, query, k=2),
        # func=lambda x: theology_tool({"question": x}, return_only_outputs=True),
        # callbacks=[StreamlitSidebarCallbackHandler()],
        description="if you can't find a linguistic answer, this is useful only for finding theological data about the user's query. Use this if the user is asking about theological concepts or value-oriented questions about 'why' the Bible says certain things. Always be sure to cite the source of the data",
    ),
    Tool(
        name="Encyclopedic Data Lookup",
        func=lambda x: encyclopedic_chroma.similarity_search(x, k=5),
        # func=lambda query: get_similar_resource(encyclopedic_chroma, query, k=2),
        # func=lambda x: encyclopedic_tool({"question": x}, return_only_outputs=True),
        # callbacks=[StreamlitSidebarCallbackHandler()],
        description="useful for finding encyclopedic data about the user's query. Use this if the user is asking about historical, cultural, geographical, archaeological, or other types of information from secondary sources",
    ),
    Tool(
        name="Any Other Kind of Question Tool",
        func=lambda x: "Sorry, I don't know!",
        description="This tool is for vague, broad, ambiguous questions",
        # callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    # human_tool,
    # Tool(
    #     name="Get Human Input Tool",
    #     func=lambda x: input(x),
    #     description="This tool is for vague, broad, ambiguous questions that require human input for clarification",
    # ),
]

# Initialize agent
mrkl = initialize_agent(
    tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# with st.form(key="form"):
#     if not enable_custom:
#         "Ask one of the sample questions, or enter your API Key in the sidebar to ask your own custom questions."
#     prefilled = st.selectbox("Sample questions", sorted(SAVED_SESSIONS.keys())) or ""
#     mrkl_input = ""

#     if enable_custom:
#         user_input = st.text_input("Or, ask your own question")
#     if not mrkl_input:
#         user_input = prefilled
#     submit_clicked = st.form_submit_button("Submit Question")

# output_container = st.empty()
# if with_clear_container(submit_clicked):
#     output_container = output_container.container()
#     output_container.chat_message("user").write(user_input)

#     answer_container = output_container.chat_message("assistant", avatar="🦜")
#     st_callback = StreamlitCallbackHandler(answer_container)

#     # If we've saved this question, play it back instead of actually running LangChain
#     # (so that we don't exhaust our API calls unnecessarily)
#     # if user_input in SAVED_SESSIONS:
#     #     session_name = SAVED_SESSIONS[user_input]
#     #     session_path = Path(__file__).parent / "runs" / session_name
#     #     print(f"Playing saved session: {session_path}")
#     #     answer = playback_callbacks([st_callback], str(session_path), max_pause_time=2)
#     # else:
#     answer = mrkl.run(user_input, callbacks=[st_callback])

#     answer_container.write(answer)
    

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = mrkl.run(prompt, callbacks=[st_callback])
        st.write(response)