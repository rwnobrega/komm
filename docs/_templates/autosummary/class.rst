{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   .. automethod:: __init__

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {%- for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block methods %}
   {%- if methods %}
   .. rubric:: Methods

   .. autosummary::
   {%- for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   .. rubric:: Documentation
