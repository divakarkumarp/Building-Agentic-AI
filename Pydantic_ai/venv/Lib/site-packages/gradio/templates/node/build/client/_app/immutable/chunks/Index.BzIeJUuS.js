import { SvelteComponent, init, safe_not_equal, create_slot, element, space, claim_element, children, claim_space, get_svelte_dataset, detach, attr, toggle_class, insert_hydration, append_hydration, listen, is_function, update_slot_base, get_all_dirty_from_scope, get_slot_changes, transition_in, transition_out, run_all, text, claim_text, set_data, noop } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
function create_if_block_1(ctx) {
  let div;
  let span;
  let t0;
  let t1;
  let t2;
  let t3;
  return {
    c() {
      div = element("div");
      span = element("span");
      t0 = text(
        /*component_type*/
        ctx[2]
      );
      t1 = text(":");
      t2 = text(" ");
      t3 = text(
        /*var_name*/
        ctx[3]
      );
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { class: true });
      var div_nodes = children(div);
      span = claim_element(div_nodes, "SPAN", { class: true });
      var span_nodes = children(span);
      t0 = claim_text(
        span_nodes,
        /*component_type*/
        ctx[2]
      );
      t1 = claim_text(span_nodes, ":");
      span_nodes.forEach(detach);
      t2 = claim_text(div_nodes, " ");
      t3 = claim_text(
        div_nodes,
        /*var_name*/
        ctx[3]
      );
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(span, "class", "svelte-1f8if9e");
      attr(div, "class", "component-name svelte-1f8if9e");
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      append_hydration(div, span);
      append_hydration(span, t0);
      append_hydration(span, t1);
      append_hydration(div, t2);
      append_hydration(div, t3);
    },
    p(ctx2, dirty) {
      if (dirty & /*component_type*/
      4)
        set_data(
          t0,
          /*component_type*/
          ctx2[2]
        );
      if (dirty & /*var_name*/
      8)
        set_data(
          t3,
          /*var_name*/
          ctx2[3]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
    }
  };
}
function create_if_block(ctx) {
  let button0;
  let textContent = "✎";
  let t1;
  let button1;
  let textContent_1 = "✗";
  let mounted;
  let dispose;
  return {
    c() {
      button0 = element("button");
      button0.textContent = textContent;
      t1 = space();
      button1 = element("button");
      button1.textContent = textContent_1;
      this.h();
    },
    l(nodes) {
      button0 = claim_element(nodes, "BUTTON", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(button0) !== "svelte-xeup1i")
        button0.textContent = textContent;
      t1 = claim_space(nodes);
      button1 = claim_element(nodes, "BUTTON", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(button1) !== "svelte-ko6iaj")
        button1.textContent = textContent_1;
      this.h();
    },
    h() {
      attr(button0, "class", "action modify svelte-1f8if9e");
      attr(button1, "class", "action delete svelte-1f8if9e");
    },
    m(target, anchor) {
      insert_hydration(target, button0, anchor);
      insert_hydration(target, t1, anchor);
      insert_hydration(target, button1, anchor);
      if (!mounted) {
        dispose = [
          listen(
            button0,
            "click",
            /*dispatch*/
            ctx[5]("modify")
          ),
          listen(
            button1,
            "click",
            /*dispatch*/
            ctx[5]("delete")
          )
        ];
        mounted = true;
      }
    },
    p: noop,
    d(detaching) {
      if (detaching) {
        detach(button0);
        detach(t1);
        detach(button1);
      }
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_fragment(ctx) {
  let div1;
  let div0;
  let show_if = (
    /*invisible_components*/
    ctx[6].includes(
      /*component_type*/
      ctx[2]
    )
  );
  let t0;
  let button0;
  let textContent = "+";
  let t2;
  let button1;
  let textContent_1 = "+";
  let t4;
  let button2;
  let textContent_2 = "+";
  let t6;
  let button3;
  let textContent_3 = "+";
  let t8;
  let t9;
  let current;
  let mounted;
  let dispose;
  let if_block0 = show_if && create_if_block_1(ctx);
  let if_block1 = !/*is_container*/
  ctx[1] && create_if_block(ctx);
  const default_slot_template = (
    /*#slots*/
    ctx[9].default
  );
  const default_slot = create_slot(
    default_slot_template,
    ctx,
    /*$$scope*/
    ctx[8],
    null
  );
  return {
    c() {
      div1 = element("div");
      div0 = element("div");
      if (if_block0)
        if_block0.c();
      t0 = space();
      button0 = element("button");
      button0.textContent = textContent;
      t2 = space();
      button1 = element("button");
      button1.textContent = textContent_1;
      t4 = space();
      button2 = element("button");
      button2.textContent = textContent_2;
      t6 = space();
      button3 = element("button");
      button3.textContent = textContent_3;
      t8 = space();
      if (if_block1)
        if_block1.c();
      t9 = space();
      if (default_slot)
        default_slot.c();
      this.h();
    },
    l(nodes) {
      div1 = claim_element(nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      div0 = claim_element(div1_nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      if (if_block0)
        if_block0.l(div0_nodes);
      t0 = claim_space(div0_nodes);
      button0 = claim_element(div0_nodes, "BUTTON", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(button0) !== "svelte-9vgc4e")
        button0.textContent = textContent;
      t2 = claim_space(div0_nodes);
      button1 = claim_element(div0_nodes, "BUTTON", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(button1) !== "svelte-pewocm")
        button1.textContent = textContent_1;
      t4 = claim_space(div0_nodes);
      button2 = claim_element(div0_nodes, "BUTTON", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(button2) !== "svelte-4vsur8")
        button2.textContent = textContent_2;
      t6 = claim_space(div0_nodes);
      button3 = claim_element(div0_nodes, "BUTTON", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(button3) !== "svelte-c945mq")
        button3.textContent = textContent_3;
      t8 = claim_space(div0_nodes);
      if (if_block1)
        if_block1.l(div0_nodes);
      div0_nodes.forEach(detach);
      t9 = claim_space(div1_nodes);
      if (default_slot)
        default_slot.l(div1_nodes);
      div1_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button0, "class", "add up svelte-1f8if9e");
      attr(button1, "class", "add left svelte-1f8if9e");
      attr(button2, "class", "add right svelte-1f8if9e");
      attr(button3, "class", "add down svelte-1f8if9e");
      attr(div0, "class", "interaction svelte-1f8if9e");
      attr(div1, "class", "sketchbox svelte-1f8if9e");
      toggle_class(
        div1,
        "row",
        /*row*/
        ctx[0]
      );
      toggle_class(
        div1,
        "active",
        /*active*/
        ctx[4]
      );
    },
    m(target, anchor) {
      insert_hydration(target, div1, anchor);
      append_hydration(div1, div0);
      if (if_block0)
        if_block0.m(div0, null);
      append_hydration(div0, t0);
      append_hydration(div0, button0);
      append_hydration(div0, t2);
      append_hydration(div0, button1);
      append_hydration(div0, t4);
      append_hydration(div0, button2);
      append_hydration(div0, t6);
      append_hydration(div0, button3);
      append_hydration(div0, t8);
      if (if_block1)
        if_block1.m(div0, null);
      append_hydration(div1, t9);
      if (default_slot) {
        default_slot.m(div1, null);
      }
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            button0,
            "click",
            /*dispatch*/
            ctx[5]("up")
          ),
          listen(
            button1,
            "click",
            /*dispatch*/
            ctx[5]("left")
          ),
          listen(
            button2,
            "click",
            /*dispatch*/
            ctx[5]("right")
          ),
          listen(
            button3,
            "click",
            /*dispatch*/
            ctx[5]("down")
          ),
          listen(div0, "click", function() {
            if (is_function(
              /*is_container*/
              ctx[1] ? void 0 : (
                /*dispatch*/
                ctx[5]("modify")
              )
            ))
              /*is_container*/
              (ctx[1] ? void 0 : (
                /*dispatch*/
                ctx[5]("modify")
              )).apply(this, arguments);
          })
        ];
        mounted = true;
      }
    },
    p(new_ctx, [dirty]) {
      ctx = new_ctx;
      if (dirty & /*component_type*/
      4)
        show_if = /*invisible_components*/
        ctx[6].includes(
          /*component_type*/
          ctx[2]
        );
      if (show_if) {
        if (if_block0) {
          if_block0.p(ctx, dirty);
        } else {
          if_block0 = create_if_block_1(ctx);
          if_block0.c();
          if_block0.m(div0, t0);
        }
      } else if (if_block0) {
        if_block0.d(1);
        if_block0 = null;
      }
      if (!/*is_container*/
      ctx[1]) {
        if (if_block1) {
          if_block1.p(ctx, dirty);
        } else {
          if_block1 = create_if_block(ctx);
          if_block1.c();
          if_block1.m(div0, null);
        }
      } else if (if_block1) {
        if_block1.d(1);
        if_block1 = null;
      }
      if (default_slot) {
        if (default_slot.p && (!current || dirty & /*$$scope*/
        256)) {
          update_slot_base(
            default_slot,
            default_slot_template,
            ctx,
            /*$$scope*/
            ctx[8],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx[8]
            ) : get_slot_changes(
              default_slot_template,
              /*$$scope*/
              ctx[8],
              dirty,
              null
            ),
            null
          );
        }
      }
      if (!current || dirty & /*row*/
      1) {
        toggle_class(
          div1,
          "row",
          /*row*/
          ctx[0]
        );
      }
      if (!current || dirty & /*active*/
      16) {
        toggle_class(
          div1,
          "active",
          /*active*/
          ctx[4]
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(default_slot, local);
      current = true;
    },
    o(local) {
      transition_out(default_slot, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div1);
      }
      if (if_block0)
        if_block0.d();
      if (if_block1)
        if_block1.d();
      if (default_slot)
        default_slot.d(detaching);
      mounted = false;
      run_all(dispose);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
  let { $$slots: slots = {}, $$scope } = $$props;
  let { row } = $$props;
  let { is_container } = $$props;
  let { component_type } = $$props;
  let { var_name } = $$props;
  let { active = false } = $$props;
  let { gradio } = $$props;
  const dispatch = (type) => {
    return () => gradio.dispatch("select", { index: 0, value: type });
  };
  const invisible_components = ["state"];
  $$self.$$set = ($$props2) => {
    if ("row" in $$props2)
      $$invalidate(0, row = $$props2.row);
    if ("is_container" in $$props2)
      $$invalidate(1, is_container = $$props2.is_container);
    if ("component_type" in $$props2)
      $$invalidate(2, component_type = $$props2.component_type);
    if ("var_name" in $$props2)
      $$invalidate(3, var_name = $$props2.var_name);
    if ("active" in $$props2)
      $$invalidate(4, active = $$props2.active);
    if ("gradio" in $$props2)
      $$invalidate(7, gradio = $$props2.gradio);
    if ("$$scope" in $$props2)
      $$invalidate(8, $$scope = $$props2.$$scope);
  };
  return [
    row,
    is_container,
    component_type,
    var_name,
    active,
    dispatch,
    invisible_components,
    gradio,
    $$scope,
    slots
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      row: 0,
      is_container: 1,
      component_type: 2,
      var_name: 3,
      active: 4,
      gradio: 7
    });
  }
}
export {
  Index as default
};
