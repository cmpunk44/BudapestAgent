{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8446e76-14f5-4c05-a050-b251167faf31",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from langchain_core.messages import HumanMessage\n",
    "from your_agent_module import budapest_agent  # importáld a meglévő LangGraph agented\n",
    "\n",
    "st.set_page_config(page_title=\"Budapest ReAct Agent\", layout=\"centered\")\n",
    "\n",
    "st.title(\"🚌 Budapest Tömegközlekedési Asszisztens\")\n",
    "st.markdown(\"Írj be egy kérdést az utazásról és látnivalókról!\")\n",
    "\n",
    "# Input mező\n",
    "user_input = st.text_input(\"Kérdés:\", placeholder=\"Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?\")\n",
    "\n",
    "if st.button(\"Küldés\") and user_input:\n",
    "    with st.spinner(\"Dolgozom a válaszon...\"):\n",
    "        msg = HumanMessage(content=user_input)\n",
    "        result = budapest_agent.graph.invoke({\"messages\": [msg]})\n",
    "        response = result['messages'][-1].content\n",
    "        st.markdown(\"### Válasz\")\n",
    "        st.write(response)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
