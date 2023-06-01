{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: Properties

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
   {% for item in (attributes + methods)|sort %}
   {%- if item in attributes %}
   .. autoproperty:: {{ item }}
   {%- else %}
   .. automethod:: {{ item }}
   {%- endif %}
   {%- endfor %}
